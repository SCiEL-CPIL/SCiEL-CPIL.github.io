import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    total_words = q ** n
    words = []
    for i in range(total_words):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        words.append(tuple(w))
        
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls
    balls_list = []
    balls_set = []
    coverers = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        w = words[i]
        covered = {i}
        for pos in range(n):
            for val in range(q):
                if val == w[pos]: continue
                new_w = list(w)
                new_w[pos] = val
                idx = word_to_idx[tuple(new_w)]
                covered.add(idx)
        balls_list.append(list(covered))
        balls_set.append(covered)
        for w_idx in covered:
            coverers[w_idx].append(i)
            
    # Initial Code: Direct Sum of Tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n-4):
            cw = h + list(tail)
            initial_code.append(word_to_idx[tuple(cw)])
            
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for c in initial_code:
        for w in balls_list[c]:
            coverage_count[w] += 1
            
    current_code = initial_code.copy()
    current_code_set = set(current_code)
    best_code = initial_code.copy()
    best_size = len(current_code)
    
    def can_remove(c):
        for w in balls_list[c]:
            if coverage_count[w] == 1:
                return False
        return True
        
    def remove_codeword(c, idx):
        for w in balls_list[c]:
            coverage_count[w] -= 1
        current_code.pop(idx)
        current_code_set.remove(c)
        
    def perform_swap(c_old, c_new, idx_old):
        for w in balls_list[c_old]:
            coverage_count[w] -= 1
        for w in balls_list[c_new]:
            coverage_count[w] += 1
        current_code[idx_old] = c_new
        current_code_set.remove(c_old)
        current_code_set.add(c_new)
        
    while time.time() < deadline:
        # Try to delete
        deleted = False
        # Randomly try a few deletions
        for _ in range(10):
            if not current_code: break
            idx = random.randint(0, len(current_code) - 1)
            c = current_code[idx]
            if can_remove(c):
                remove_codeword(c, idx)
                deleted = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = current_code.copy()
                if time.time() > deadline: break
        if time.time() > deadline: break
        
        if not deleted:
            # Try swaps
            if current_code:
                idx_old = random.randint(0, len(current_code) - 1)
                c_old = current_code[idx_old]
                
                # Find critical words
                s_crit = [w for w in balls_list[c_old] if coverage_count[w] == 1]
                
                if not s_crit:
                    # Should have been deleted, but if not, force delete?
                    # This case implies can_remove returned False erroneously or state changed
                    # Actually, if s_crit is empty, can_remove would be True.
                    # So this block is effectively unreachable if can_remove is correct.
                    pass
                else:
                    # Pick a random critical word
                    w_crit = random.choice(s_crit)
                    # Pick a random codeword covering it
                    candidates = coverers[w_crit]
                    c_new = -1
                    for _ in range(20):
                        c_candidate = random.choice(candidates)
                        if c_candidate not in current_code_set:
                            c_new = c_candidate
                            break
                    
                    if c_new != -1:
                        # Check if c_new covers all s_crit
                        valid = True
                        for w in s_crit:
                            if w not in balls_set[c_new]:
                                valid = False
                                break
                        
                        if valid:
                            perform_swap(c_old, c_new, idx_old)
                            
    return [list(words[c]) for c in best_code]
