import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute place values
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
    
    # Precompute all words
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q
    
    # Precompute balls as a 2D array
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for i in range(total):
        w = words[i]
        balls[i, 0] = i
        idx = 1
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                balls[i, idx] = new_idx
                idx += 1
    
    # Precompute ball sets for O(1) membership in swap validation
    ball_sets = [set(balls[i].tolist()) for i in range(total)]
    
    def greedy_build(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        covered = np.zeros(total, dtype=bool)
        code = []
        
        while not np.all(covered):
            new_cov = np.sum(~covered[balls], axis=1)
            max_cov = np.max(new_cov)
            if max_cov == 0:
                break
            candidates = np.where(new_cov == max_cov)[0]
            if randomize and len(candidates) > 1:
                best_idx = random.choice(candidates.tolist())
            else:
                best_idx = candidates[0]
            code.append(best_idx)
            covered[balls[best_idx]] = True
        return code
    
    def reduce_code(code_set, cnt):
        """Remove redundant codewords. Modifies code_set and cnt in place."""
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                if np.min(cnt[balls[c]]) >= 2:
                    code_set.remove(c)
                    cnt[balls[c]] -= 1
                    changed = True
    
    def try_swap(c_remove, c_add, code_set, cnt):
        """Try swap: remove c_remove, add c_add. Returns True if successful."""
        if c_add in code_set:
            return False
        if c_remove == c_add:
            return False
        
        ball_rem = balls[c_remove]
        ball_add = balls[c_add]
        set_add = ball_sets[c_add]
        
        valid = True
        for x in ball_rem:
            if x not in set_add:
                if cnt[x] < 2:
                    valid = False
                    break
        
        if not valid:
            return False
        
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt[ball_rem] -= 1
        cnt[ball_add] += 1
        return True
    
    best_code = None
    best_size = total
    
    restarts = 0
    while time.time() < deadline - 8.0:
        restarts += 1
        if restarts % 3 == 0:
            code = greedy_build(randomize=True, seed=restarts)
        else:
            code = greedy_build(randomize=False)
        
        code_set = set(code)
        
        # Build counts
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[balls[c]] += 1
        
        # Delete phase
        reduce_code(code_set, cnt)
        
        current_size = len(code_set)
        if current_size < best_size:
            best_size = current_size
            best_code = list(code_set)
        
        # Swap-based local search
        if time.time() < deadline - 5.0 and current_size > 156:
            max_swaps = 300
            swaps_attempted = 0
            while swaps_attempted < max_swaps and time.time() < deadline - 5.0:
                swaps_attempted += 1
                if len(code_set) == 0:
                    break
                
                c = random.choice(list(code_set))
                c_prime = random.randint(0, total - 1)
                
                if try_swap(c, c_prime, code_set, cnt):
                    # After successful swap, run delete phase
                    reduce_code(code_set, cnt)
                    current_size = len(code_set)
                    if current_size < best_size:
                        best_size = current_size
                        best_code = list(code_set)
                    if best_size <= 160:
                        break
            
            # Final delete after swap phase
            reduce_code(code_set, cnt)
            current_size = len(code_set)
            if current_size < best_size:
                best_size = current_size
                best_code = list(code_set)
        
        if best_size <= 160:
            break
    
    # Final reduction
    if best_code is not None and time.time() < deadline - 3.0:
        code_set = set(best_code)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[balls[c]] += 1
        reduce_code(code_set, cnt)
        best_size = len(code_set)
        best_code = list(code_set)
    
    # Convert indices back to words
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
        
    return result
