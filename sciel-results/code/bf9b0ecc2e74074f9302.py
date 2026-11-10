"""Construct signs to minimize max absolute sum on any arithmetic progression in [1,256]."""
import math
import random
import time
import numpy as np

def construct():
    n = 256
    
    # Precompute AP indices? No, we use the step-based vectorization.
    
    # Optimized discrepancy calculation
    def compute_max_discrepancy_and_worst(s):
        max_abs = 1
        worst_aps = []
        
        for step in range(1, n + 1):
            max_len = (n - 1) // step + 1
            # Create data array
            # We can avoid the inner loop for filling if we use advanced indexing?
            # But the loop is small (max 256).
            data = np.zeros((step, max_len), dtype=np.int8)
            for i in range(step):
                seq = s[i::step]
                data[i, :len(seq)] = seq
            
            csum = np.cumsum(data, axis=1)
            abs_csum = np.abs(csum)
            
            # Max per row
            row_maxes = np.max(abs_csum, axis=1)
            local_max = np.max(row_maxes)
            
            if local_max > max_abs:
                max_abs = local_max
                worst_rows = np.where(row_maxes == local_max)[0]
                worst_aps = [(step, r) for r in worst_rows]
            elif local_max == max_abs:
                worst_rows = np.where(row_maxes == local_max)[0]
                worst_aps.extend([(step, r) for r in worst_rows])
                
        return max_abs, worst_aps

    def is_prime(x):
        if x < 2:
            return False
        if x % 2 == 0:
            return x == 2
        if x % 3 == 0:
            return x == 3
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True

    # Step 1: Generate multiple initial candidates
    candidates = []
    
    # Legendre symbols for primes near n
    primes = []
    for p in range(max(2, n - 50), n + 50):
        if is_prime(p):
            primes.append(p)
            
    for p in primes[:10]:
        legendre = []
        for a in range(1, n + 1):
            if a % p == 0:
                legendre.append(1)
            else:
                val = pow(a, (p - 1) // 2, p)
                if val == 1:
                    legendre.append(1)
                else:
                    legendre.append(-1)
        candidates.append(np.array(legendre, dtype=np.int8))
        
    # Thue-Morse sequence
    tm = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Popcount of i
        # tm[i] = 1 if popcount is even, -1 if odd?
        # Standard Thue-Morse is 0/1. We map to -1/1.
        # t_i = (-1)^{popcount(i)}
        pc = bin(i).count('1')
        tm[i] = 1 if pc % 2 == 0 else -1
    candidates.append(tm)
    
    # Random candidates
    rng = random.Random(42)
    for _ in range(5):
        candidates.append(np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8))

    # Evaluate all candidates and pick the best
    best_signs = None
    best_max = float('inf')
    
    for cand in candidates:
        curr_max, _ = compute_max_discrepancy_and_worst(cand)
        if curr_max < best_max:
            best_max = curr_max
            best_signs = cand.copy()

    signs = best_signs.copy()
    current_max = best_max

    # Step 2: Iterated Local Search
    start_time = time.time()
    time_limit = 55.0
    
    def local_search(s, current_max):
        improved = True
        max_iter = 50
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Find worst APs
            max_val, worst_aps = compute_max_discrepancy_and_worst(s)
            if max_val < current_max:
                current_max = max_val
                s = s.copy() # Ensure we work on a copy if we modify? 
                # Actually, we modify s in place.
                
            if not worst_aps:
                break
                
            # Shuffle worst APs to randomize
            rng.shuffle(worst_aps)
            
            # Try flipping signs in the top few worst APs
            flip_found = False
            # Limit to 10 APs to save time
            for step, start in worst_aps[:10]:
                idx = np.arange(start, n, step)
                # Shuffle indices within the AP
                idx = idx[np.random.permutation(len(idx))]
                
                for i in idx:
                    old_val = s[i]
                    s[i] = -old_val
                    
                    new_max, _ = compute_max_discrepancy_and_worst(s)
                    
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
                        flip_found = True
                        break
                    else:
                        s[i] = old_val
                        
                if flip_found:
                    break
            
            if not flip_found:
                break
                
        return s, current_max

    # Main ILS loop
    no_improve_count = 0
    max_no_improve = 3
    
    while time.time() - start_time < time_limit:
        signs, current_max = local_search(signs, current_max)
        
        if current_max < best_max:
            best_max = current_max
            best_signs = signs.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= max_no_improve:
            # Perturb
            trial = best_signs.copy()
            num_flips = rng.randint(5, 15)
            flip_indices = rng.sample(range(n), num_flips)
            for i in flip_indices:
                trial[i] = -trial[i]
            
            signs = trial
            current_max, _ = compute_max_discrepancy_and_worst(signs)
            no_improve_count = 0

    return [int(x) for x in best_signs]
