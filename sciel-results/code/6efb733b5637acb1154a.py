import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    # Generate primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Window size
    W = 30000
    
    # Initialize residues randomly
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    def update_coverage(cov, i, old_r, new_r):
        p = primes_arr[i]
        old_start = old_r if old_r >= 1 else p
        if old_start <= W:
            cov[old_start:W+1:p] -= 1
        new_start = new_r if new_r >= 1 else p
        if new_start <= W:
            cov[new_start:W+1:p] += 1

    def get_max_L(cov):
        bad = np.where(cov[1:W+1] < 2)[0]
        if len(bad) == 0:
            return W
        return bad[0]
        
    cov = np.zeros(W + 1, dtype=np.int32)
    for i in range(n_primes):
        p = primes_arr[i]
        r = residues[i]
        start = r if r >= 1 else p
        if start <= W:
            cov[start:W+1:p] += 1
            
    current_L = get_max_L(cov)
    best_L = current_L
    best_residues = residues.copy()
    
    T = 5.0
    cooling = 0.9995
    
    # Pre-compute indices for speed? No, slicing is fast enough.
    
    while time.time() < deadline - 2.0:
        # Targeted Move
        if current_L < W:
            gap_pos = current_L + 1
            gap_cov = cov[gap_pos]
            
            # Find primes that do NOT cover gap_pos
            # We want to change one (or two) to cover it.
            # To avoid O(N) scan every time, we can just pick a few random primes
            # and check if they cover gap_pos.
            
            # But scanning 168 primes is fast.
            candidates = []
            for i in range(n_primes):
                p = primes_arr[i]
                if residues[i] != gap_pos % p:
                    candidates.append(i)
            
            if candidates:
                if gap_cov == 0:
                    if len(candidates) >= 2:
                        i1, i2 = random.sample(candidates, 2)
                        old_r1 = residues[i1]
                        old_r2 = residues[i2]
                        new_r1 = gap_pos % primes_arr[i1]
                        new_r2 = gap_pos % primes_arr[i2]
                        
                        update_coverage(cov, i1, old_r1, new_r1)
                        update_coverage(cov, i2, old_r2, new_r2)
                        residues[i1] = new_r1
                        residues[i2] = new_r2
                        
                        new_L = get_max_L(cov)
                        
                        if new_L > current_L:
                            current_L = new_L
                            if current_L > best_L:
                                best_L = current_L
                                best_residues = residues.copy()
                        else:
                            update_coverage(cov, i2, new_r2, old_r2)
                            update_coverage(cov, i1, new_r1, old_r1)
                            residues[i1] = old_r1
                            residues[i2] = old_r2
                            
                elif gap_cov == 1:
                    i1 = random.choice(candidates)
                    old_r1 = residues[i1]
                    new_r1 = gap_pos % primes_arr[i1]
                    
                    update_coverage(cov, i1, old_r1, new_r1)
                    residues[i1] = new_r1
                    
                    new_L = get_max_L(cov)
                    
                    if new_L > current_L:
                        current_L = new_L
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues.copy()
                    else:
                        update_coverage(cov, i1, new_r1, old_r1)
                        residues[i1] = old_r1
                        
        # Random Move
        i = random.randint(0, n_primes - 1)
        p = primes_arr[i]
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
            
        update_coverage(cov, i, old_r, new_r)
        residues[i] = new_r
        
        new_L = get_max_L(cov)
        
        delta_L = new_L - current_L
        
        if delta_L > 0 or random.random() < np.exp(delta_L / T):
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
        else:
            update_coverage(cov, i, new_r, old_r)
            residues[i] = old_r
            
        T *= cooling
        if T < 0.1:
            T = 0.1

    # Final Polish: Try to fix the last gap with all primes
    if time.time() < deadline - 1.0 and current_L < W:
        gap_pos = current_L + 1
        for i in range(n_primes):
            if time.time() > deadline - 1.0:
                break
            p = primes_arr[i]
            old_r = residues[i]
            new_r = gap_pos % p
            if new_r == old_r:
                continue
            update_coverage(cov, i, old_r, new_r)
            residues[i] = new_r
            new_L = get_max_L(cov)
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                update_coverage(cov, i, new_r, old_r)
                residues[i] = old_r

    return [int(r) for r in best_residues]
