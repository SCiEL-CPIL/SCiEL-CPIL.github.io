"""Construct signs to minimize max absolute sum on any arithmetic progression in [1,256]."""
import math
import random
import time
import numpy as np

def construct():
    n = 256
    
    # Step 1: Construct initial sequence using Legendre symbol modulo a prime near n
    # Find the smallest prime p >= n
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
    
    p = n
    while not is_prime(p):
        p += 1
    
    # Legendre symbol (a/p) for a = 1..n
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
    
    signs = np.array(legendre, dtype=np.int8)
    
    # Step 2: Fast discrepancy calculation using NumPy
    def compute_max_discrepancy(s):
        n_local = len(s)
        max_abs = 0
        for step in range(1, n_local + 1):
            for r in range(step):
                subseq = s[r::step]
                if len(subseq) == 0:
                    continue
                csum = np.cumsum(subseq)
                local_max = np.max(np.abs(csum))
                if local_max > max_abs:
                    max_abs = local_max
        return max_abs

    current_max = compute_max_discrepancy(signs)
    
    best_signs = signs.copy()
    best_max = current_max
    
    # Step 3: Local Search with NumPy speedup
    start_time = time.time()
    time_limit = 50.0
    
    rng = random.Random(42)
    
    # First-improvement hill climbing with random restarts
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() - start_time < time_limit:
        improved = False
        # Shuffle order of indices to avoid bias
        indices = list(range(n))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > time_limit:
                break
            
            # Flip sign at i
            old_val = signs[i]
            signs[i] = -old_val
            
            new_max = compute_max_discrepancy(signs)
            
            if new_max < current_max:
                # Accept improvement
                current_max = new_max
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Revert
                signs[i] = old_val
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Perturb current best
                trial = best_signs.copy()
                for _ in range(5):
                    idx = rng.randint(0, n - 1)
                    trial[idx] = -trial[idx]
                
                signs = trial
                current_max = compute_max_discrepancy(signs)
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
                
                no_improve_count = 0
        else:
            no_improve_count = 0

    return [int(x) for x in best_signs]
