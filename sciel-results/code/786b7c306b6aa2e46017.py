import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other cases
        code = []
        covered = set()
        all_words = list(itertools.product(range(q), repeat=n))
        random.shuffle(all_words)
        for w in all_words:
            is_covered = False
            for c in code:
                dist = sum(1 for a, b in zip(w, c) if a != b)
                if dist <= radius:
                    is_covered = True
                    break
            if not is_covered:
                code.append(w)
        return code

    total_words = q ** n
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors and neighbor sets for fast lookup
    neighbors = []
    neighbor_sets = []
    for i in range(total_words):
        w = idx_to_word(i)
        covered = [i]
        for pos in range(n):
            original = w[pos]
            for val in range(q):
                if val == original:
                    continue
                w[pos] = val
                covered.append(word_to_idx(w))
            w[pos] = original
        neighbors.append(covered)
        neighbor_sets.append(set(covered))
        
    # Initial code: Tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_words = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code_words.append(h + list(tail))
            
    initial_indices = set(word_to_idx(w) for w in initial_code_words)
    
    code_set = set(initial_indices)
    in_code = np.zeros(total_words, dtype=bool)
    for i in code_set:
        in_code[i] = True
        
    count = np.zeros(total_words, dtype=np.int32)
    for i in code_set:
        for j in neighbors[i]:
            count[j] += 1
            
    def greedy_removal():
        changed = True
        while changed:
            changed = False
            items = list(code_set)
            random.shuffle(items)
            for i in items:
                if i not in code_set:
                    continue
                removable = True
                for j in neighbors[i]:
                    if count[j] == 1:
                        removable = False
                        break
                if removable:
                    code_set.remove(i)
                    in_code[i] = False
                    for j in neighbors[i]:
                        count[j] -= 1
                    changed = True

    greedy_removal()
    
    start_time = time.time()
    time_limit = 115.0
    
    best_size = len(code_set)
    best_code = set(code_set)
    stuck_count = 0
    
    while time.time() - start_time < time_limit:
        if not code_set:
            break
            
        move_type = random.random()
        
        if move_type < 0.5:
            # Try removal or swap
            i = random.choice(list(code_set))
            critical = False
            U = []
            for j in neighbors[i]:
                if count[j] == 1:
                    critical = True
                    U.append(j)
            
            if not critical:
                code_set.remove(i)
                in_code[i] = False
                for j in neighbors[i]:
                    count[j] -= 1
                greedy_removal()
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = set(code_set)
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                found_k = -1
                for k in range(total_words):
                    if in_code[k]:
                        continue
                    covers_all = True
                    for u in U:
                        if u not in neighbor_sets[k]:
                            covers_all = False
                            break
                    if covers_all:
                        found_k = k
                        break
                
                if found_k != -1:
                    code_set.remove(i)
                    in_code[i] = False
                    for j in neighbors[i]:
                        count[j] -= 1
                    code_set.add(found_k)
                    in_code[found_k] = True
                    for j in neighbors[found_k]:
                        count[j] += 1
                    greedy_removal()
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = set(code_set)
                        stuck_count = 0
                    else:
                        stuck_count += 1
        elif move_type < 0.9:
            # Add and Reduce with Backtracking
            # Pick a random word not in code
            while True:
                k = random.randint(0, total_words - 1)
                if not in_code[k]:
                    break
            
            old_code_set = set(code_set)
            old_count = count.copy()
            
            code_set.add(k)
            in_code[k] = True
            for j in neighbors[k]:
                count[j] += 1
            greedy_removal()
            
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = set(code_set)
                stuck_count = 0
            else:
                # Revert
                code_set = old_code_set
                in_code[:] = False
                for i in code_set:
                    in_code[i] = True
                count[:] = old_count
                stuck_count += 1
        else:
            # Restart from best if stuck
            if stuck_count > 1000:
                code_set = set(best_code)
                in_code[:] = False
                for i in code_set:
                    in_code[i] = True
                count[:] = 0
                for i in code_set:
                    for j in neighbors[i]:
                        count[j] += 1
                greedy_removal()
                stuck_count = 0
            else:
                stuck_count += 1

    final_indices = list(best_code)
    final_words = [idx_to_word(i) for i in final_indices]
    return final_words
