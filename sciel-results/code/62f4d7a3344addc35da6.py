import numpy as np
import time
import itertools
import random
from collections import defaultdict

def construct(q, n, radius):
    """
    Constructs a covering code using:
    1. Greedy Set Cover with randomization
    2. Aggressive Local Search with Pruning, Swaps, and Add-Remove moves
    3. Final Polish with intensive swap attempts
    """
    start_time = time.time()
    timeout = 118

    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(word) for word in itertools.product(range(q), repeat=n)]

    total_words = 3 ** 7
    n_words = total_words

    # Precompute digits and balls
    digits = np.zeros((n_words, n), dtype=int)
    for i in range(n_words):
        temp = i
        for k in range(n):
            digits[i, k] = temp % 3
            temp //= 3

    # Precompute balls as lists of indices
    balls = []
    powers3 = [1, 3, 9, 27, 81, 243, 729]
    for i in range(n_words):
        ball = [i]
        for pos in range(n):
            orig_digit = digits[i, pos]
            p = powers3[pos]
            base_val = i - orig_digit * p
            for new_digit in range(3):
                if new_digit == orig_digit:
                    continue
                j = base_val + new_digit * p
                ball.append(j)
        balls.append(ball)

    # Precompute ball sets for faster membership testing
    ball_sets = [set(b) for b in balls]
    
    # Precompute ball membership as a boolean matrix: ball_matrix[i, j] = True if j in balls[i]
    ball_matrix = np.zeros((n_words, n_words), dtype=bool)
    for i in range(n_words):
        for j in balls[i]:
            ball_matrix[i, j] = True

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_words, dtype=bool)
        code = []
        
        while np.any(uncovered):
            if time.time() - start_time > timeout:
                break
            
            best_idx = -1
            max_cov = 0
            best_candidates = []
            
            for i in range(n_words):
                cov = 0
                for j in balls[i]:
                    if uncovered[j]:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_candidates = [i]
                    if max_cov == 15:
                        break
                elif cov == max_cov and max_cov > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break
            best_idx = rng.choice(best_candidates)
            code.append(best_idx)
            for j in balls[best_idx]:
                uncovered[j] = False
                
        return code

    def local_search(code, seed, max_swap_attempts=500):
        if time.time() - start_time > timeout:
            return code
            
        count = np.zeros(n_words, dtype=int)
        for c in code:
            count[balls[c]] += 1
                
        code_set = set(code)
        rng = random.Random(seed)
        
        def prune():
            nonlocal code, count, code_set
            new_code = []
            changed = False
            for c in code:
                redundant = True
                for j in balls[c]:
                    if count[j] < 2:
                        redundant = False
                        break
                if redundant:
                    count[balls[c]] -= 1
                    code_set.discard(c)
                    changed = True
                else:
                    new_code.append(c)
            if changed:
                code = new_code
            return changed

        prune()
        
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            
            if prune():
                improved = True
                continue
            
            swap_done = False
            for _ in range(max_swap_attempts):
                if time.time() - start_time > timeout:
                    return code
                if len(code) == 0:
                    break
                
                c_idx = rng.randint(0, len(code) - 1)
                c = code[c_idx]
                
                # Find critical positions for c: positions j where count[j] < 2
                critical = [j for j in balls[c] if count[j] < 2]
                
                found_swap = False
                if critical:
                    # For each critical position, we need w to cover it
                    # Precompute: for each position j, which words cover j
                    # Use ball_matrix to find candidates quickly
                    for j in critical:
                        # Find all words w that cover j and are not in code_set
                        candidates = np.where(ball_matrix[:, j])[0]
                        for w in candidates:
                            w = int(w)
                            if w == c or w in code_set:
                                continue
                            # Check if w covers all critical positions
                            valid = True
                            for cj in critical:
                                if not ball_matrix[w, cj]:
                                    valid = False
                                    break
                            if valid:
                                count[balls[c]] -= 1
                                count[balls[w]] += 1
                                code[c_idx] = w
                                code_set.discard(c)
                                code_set.add(w)
                                improved = True
                                swap_done = True
                                found_swap = True
                                break
                        if found_swap:
                            break
                
                if found_swap:
                    break
            
            if swap_done:
                continue
            
            if len(code) < n_words:
                w = rng.randint(0, n_words - 1)
                if w not in code_set:
                    old_code = list(code)
                    old_count = count.copy()
                    old_set = set(code_set)
                    
                    code.append(w)
                    code_set.add(w)
                    count[balls[w]] += 1
                    
                    prune()
                    
                    if len(code) < len(old_code):
                        improved = True
                    else:
                        code = old_code
                        count = old_count
                        code_set = old_set
                        
        return code

    def polish(code, seed, time_budget):
        if time.time() - start_time > timeout:
            return code
        
        count = np.zeros(n_words, dtype=int)
        for c in code:
            count[balls[c]] += 1
                
        code_set = set(code)
        rng = random.Random(seed)
        
        def prune():
            nonlocal code, count, code_set
            new_code = []
            changed = False
            for c in code:
                redundant = True
                for j in balls[c]:
                    if count[j] < 2:
                        redundant = False
                        break
                if redundant:
                    count[balls[c]] -= 1
                    code_set.discard(c)
                    changed = True
                else:
                    new_code.append(c)
            if changed:
                code = new_code
            return changed
        
        prune()
        
        attempts = 0
        while time.time() - start_time < time_budget:
            if len(code) == 0:
                break
            
            c_idx = rng.randint(0, len(code) - 1)
            c = code[c_idx]
            
            critical = [j for j in balls[c] if count[j] < 2]
            
            found_swap = False
            if critical:
                for j in critical:
                    candidates = np.where(ball_matrix[:, j])[0]
                    for w in candidates:
                        w = int(w)
                        if w == c or w in code_set:
                            continue
                        valid = True
                        for cj in critical:
                            if not ball_matrix[w, cj]:
                                valid = False
                                break
                        if valid:
                            count[balls[c]] -= 1
                            count[balls[w]] += 1
                            code[c_idx] = w
                            code_set.discard(c)
                            code_set.add(w)
                            prune()
                            found_swap = True
                            break
                    if found_swap:
                        break
            
            if found_swap:
                attempts += 1
            
            if attempts % 100 == 0 and time.time() - start_time > timeout - 5:
                break
        
        return code

    best_code = None
    best_size = float('inf')

    num_trials = 30
    for trial in range(num_trials):
        if time.time() - start_time > timeout - 20:
            break
            
        seed = 100 + trial
        code = greedy_cover(seed)
        
        if len(code) < best_size:
            best_code = list(code)
            best_size = len(code)
            
        code = local_search(code, seed, max_swap_attempts=800)
        
        if len(code) < best_size:
            best_code = list(code)
            best_size = len(code)
            
        remaining = timeout - (time.time() - start_time)
        if best_size <= 200 and remaining > 25:
            extra_time = timeout - 15
            code = local_search(list(best_code), seed + 1000, max_swap_attempts=500)
            if len(code) < best_size:
                best_code = list(code)
                best_size = len(code)

    remaining = timeout - (time.time() - start_time)
    if best_code is not None and remaining > 10:
        polish_time = min(remaining - 5, 50)
        best_code = polish(best_code, 99999, start_time + polish_time + 5)
        best_size = len(best_code)
    
    if best_code is None:
        best_code = list(range(n_words))
        
    result = [list(digits[c]) for c in best_code]
    return result
