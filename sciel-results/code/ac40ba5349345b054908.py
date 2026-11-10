"""Construct signs to minimize max absolute sum on any arithmetic progression in [1,256]."""
import math
import random
import time
import numpy as np

def construct():
    n = 256
    
    # Precompute all arithmetic progressions as index arrays
    # This allows faster access in the discrepancy calculation
    ap_indices = []
    for step in range(1, n + 1):
        for start in range(step):
            idx = list(range(start, n, step))
            if idx:
                ap_indices.append(np.array(idx, dtype=np.int32))
    
    # Optimized discrepancy calculation with early exit
    def compute_max_discrepancy(s, threshold=None):
        max_abs = 0
        for idx in ap_indices:
            # Cast to int32 to ensure safe cumulative sum
            subseq = s[idx].astype(np.int32)
            csum = np.cumsum(subseq)
            local_max = np.max(np.abs(csum))
            
            if local_max > max_abs:
                max_abs = local_max
                if threshold is not None and max_abs >= threshold:
                    return max_abs
        return max_abs

    # Step 1: Generate multiple initial candidates
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

    candidates = []
    p = n
    count_primes = 0
    while count_primes < 10:
        if is_prime(p):
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
            count_primes += 1
        p += 1
        
    # Add a random candidate
    rng = random.Random(42)
    candidates.append(np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8))

    # Evaluate all candidates and pick the best
    best_signs = None
    best_max = float('inf')
    
    for cand in candidates:
        curr_max = compute_max_discrepancy(cand)
        if curr_max < best_max:
            best_max = curr_max
            best_signs = cand.copy()

    signs = best_signs
    current_max = best_max

    # Step 2: Iterated Local Search
    start_time = time.time()
    time_limit = 55.0
    
    # Local search function: First improvement hill climbing
    def local_search(s, current_max):
        improved = True
        while improved:
            improved = False
            indices = list(range(n))
            rng.shuffle(indices)
            
            for i in indices:
                old_val = s[i]
                s[i] = -old_val
                # Use threshold for early exit
                new_max = compute_max_discrepancy(s, current_max)
                
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                else:
                    s[i] = old_val
        return s, current_max

    # Main ILS loop
    no_improve_count = 0
    max_no_improve = 10
    
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
            num_flips = rng.randint(3, 7)
            flip_indices = rng.sample(range(n), num_flips)
            for i in flip_indices:
                trial[i] = -trial[i]
            
            signs = trial
            current_max = compute_max_discrepancy(signs)
            no_improve_count = 0

    return [int(x) for x in best_signs]
