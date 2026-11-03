"""Ternary covering code K_3(8,1) via vectorized local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # 17 for q=3, n=8, radius=1

    # Precompute balls: list of numpy arrays of indices
    balls = []
    for i in range(total_words):
        # Convert i to base-q digits
        digits = []
        tmp = i
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        
        ball = [i]
        for pos in range(n):
            original_val = digits[pos]
            for val in range(q):
                if val == original_val:
                    continue
                new_idx = i + (val - original_val) * (q ** (n - 1 - pos))
                ball.append(new_idx)
        balls.append(np.array(ball, dtype=np.int32))

    def greedy_init():
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        coverage = np.zeros(total_words, dtype=np.int32)
        
        max_iters = total_words * 2
        iter_count = 0
        while np.any(uncovered) and iter_count < max_iters and time.time() < deadline - 60:
            iter_count += 1
            w = random.randint(0, total_words - 1)
            if not uncovered[w]:
                continue
                
            best_c = -1
            best_score = -1
            for c in balls[w]:
                score = np.sum(uncovered[balls[c]])
                if score > best_score:
                    best_score = score
                    best_c = c
            
            if best_c == -1:
                continue

            code.append(best_c)
            coverage[balls[best_c]] += 1
            uncovered[balls[best_c]] = (coverage[balls[best_c]] == 0)

        return code, coverage

    best_code = None
    best_len = float('inf')
    
    num_restarts = 10
    for _ in range(num_restarts):
        if time.time() > deadline - 60:
            break
        code, coverage = greedy_init()
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]
            best_coverage = coverage.copy()
            
    if best_code is None:
        code, coverage = greedy_init()
        best_code = code
        best_coverage = coverage
        best_len = len(code)

    code = best_code
    coverage = best_coverage.copy()
    code_set = set(code)

    def try_remove(c):
        nonlocal coverage, code, code_set
        b = balls[c]
        if np.min(coverage[b]) > 1:
            coverage[b] -= 1
            code.remove(c)
            code_set.remove(c)
            return True
        return False

    # Initial removal pass
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        random.shuffle(code)
        for c in code[:]:
            if try_remove(c):
                improved = True

    # Local search with swaps
    while time.time() < deadline:
        if len(code) < 2:
            break
            
        idx1 = random.randint(0, len(code) - 1)
        c1 = code[idx1]
        b1 = balls[c1]
        
        if np.min(coverage[b1]) > 1:
            try_remove(c1)
            continue
            
        crit_mask = (coverage[b1] == 1)
        if not np.any(crit_mask):
            continue
            
        critical_words = b1[crit_mask]
        w_crit = critical_words[random.randint(0, len(critical_words) - 1)]
        
        candidates = balls[w_crit]
        
        code_backup = code.copy()
        coverage_backup = coverage.copy()
        code_set_backup = code_set.copy()
        
        valid_c2 = None
        for c2 in candidates:
            if c2 in code_set:
                continue
            b2 = balls[c2]
            is_subset = True
            for w in critical_words:
                if w not in b2:
                    is_subset = False
                    break
            if is_subset:
                valid_c2 = c2
                break
                
        if valid_c2 is not None:
            # Perform swap
            coverage[b1] -= 1
            coverage[balls[valid_c2]] += 1
            code[idx1] = valid_c2
            code_set.remove(c1)
            code_set.add(valid_c2)
            
            removed = False
            for c in code[:]:
                if try_remove(c):
                    removed = True
                    break
                    
            if not removed:
                # Revert
                code = code_backup
                coverage = coverage_backup
                code_set = code_set_backup

    # Final removal pass
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(code)
        for c in code[:]:
            if try_remove(c):
                improved = True

    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))

    result = [list(idx_to_tuple(c)) for c in code]
    return result

# EVOLVE-BLOCK-END
