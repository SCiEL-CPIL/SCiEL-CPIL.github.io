import numpy as np
import time
import math

def construct():
    # Primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    L_max = 2000
    
    # Initial residues: random
    residues = np.array([np.random.randint(0, p) for p in primes])
    
    # Initialize counts
    counts = np.zeros(L_max + 1, dtype=np.int8)
    
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            idx = np.arange(p, L_max + 1, p)
        else:
            idx = np.arange(r, L_max + 1, p)
        if len(idx) > 0:
            np.add.at(counts, idx, 1)
            
    def get_L(c):
        # c is counts array
        fail_idx = np.where(c[1:] < 2)[0]
        if len(fail_idx) == 0:
            return L_max
        else:
            return fail_idx[0]
            
    current_L = get_L(counts)
    best_L = current_L
    best_res = residues.copy()
    
    T = 5.0
    cooling_rate = 0.9995
    min_T = 0.1
    
    while time.time() < deadline:
        i = np.random.randint(n_primes)
        p = primes[i]
        
        old_r = residues[i]
        new_r = np.random.randint(0, p)
        
        if new_r == old_r:
            continue
            
        # Prepare indices
        if old_r == 0:
            idx_old = np.arange(p, L_max + 1, p)
        else:
            idx_old = np.arange(old_r, L_max + 1, p)
            
        if new_r == 0:
            idx_new = np.arange(p, L_max + 1, p)
        else:
            idx_new = np.arange(new_r, L_max + 1, p)
            
        # Apply change
        if len(idx_old) > 0:
            counts[idx_old] -= 1
        if len(idx_new) > 0:
            counts[idx_new] += 1
            
        new_L = get_L(counts)
        
        if new_L > current_L:
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_res = residues.copy()
                # Optional: if we hit L_max, we could try to extend, but let's keep it simple
        else:
            delta = new_L - current_L
            accept = False
            if delta == 0:
                # Accept equal moves with some probability to explore
                accept = np.random.rand() < 0.5
            elif delta < 0:
                if T > min_T:
                    accept = np.random.rand() < math.exp(delta / T)
                else:
                    accept = False
                    
            if not accept:
                # Revert
                if len(idx_new) > 0:
                    counts[idx_new] -= 1
                if len(idx_old) > 0:
                    counts[idx_old] += 1
                residues[i] = old_r
            else:
                residues[i] = new_r
                current_L = new_L # update current state L
                
        T *= cooling_rate
        if T < min_T:
            T = min_T

    # Final verification and potential extension
    final_res = best_res
    final_L = best_L
    
    # If we reached L_max, check if we can go further
    if final_L >= L_max:
        L_check = L_max * 2
        counts_check = np.zeros(L_check + 1, dtype=np.int8)
        for i, p in enumerate(primes):
            r = final_res[i]
            if r == 0:
                idx = np.arange(p, L_check + 1, p)
            else:
                idx = np.arange(r, L_check + 1, p)
            if len(idx) > 0:
                np.add.at(counts_check, idx, 1)
                
        fail_idx = np.where(counts_check[1:] < 2)[0]
        if len(fail_idx) == 0:
            final_L = L_check
        else:
            final_L = fail_idx[0]
            
    return [int(x) for x in final_res]
