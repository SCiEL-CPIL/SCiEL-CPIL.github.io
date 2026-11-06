import numpy as np
import time
import random

def construct():
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    L_MAX = 20000 # Search a bit further
    counts = np.zeros(L_MAX + 1, dtype=np.int16)
    
    # Heuristic Init:
    # Set residues to cover small numbers densely.
    # Try to cover 1, 2, ... with pairs.
    # Simple random init is okay if SA is strong.
    residues = np.array([random.randint(0, p-1) for p in primes], dtype=np.int64)
    
    for i, p in enumerate(primes):
        r = residues[i]
        counts[r:L_MAX+1:p] += 1
        
    def get_L():
        fail = counts[1:] < 2
        if not np.any(fail):
            return L_MAX
        idx = np.argmax(fail)
        return idx
        
    current_L = get_L()
    best_L = current_L
    best_residues = residues.copy()
    
    start_time = time.time()
    time_limit = 115.0
    
    # Phase 1: Quick Greedy / Local Search
    # Try to improve L by changing one residue at a time to the best option.
    # This is expensive if done for all residues.
    # Instead, just run SA with a high temperature for a short time.
    
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() - start_time < time_limit:
        # Move Selection
        # 50% chance: Target the first gap
        # 50% chance: Random
        if random.random() < 0.5:
            # Find first gap
            fail = counts[1:] < 2
            if np.any(fail):
                gap_idx = np.argmax(fail)
                x_gap = gap_idx + 1
                
                # Pick a random prime
                i = random.randint(0, n_primes - 1)
                p = primes[i]
                r_new = x_gap % p
            else:
                i = random.randint(0, n_primes - 1)
                p = primes[i]
                r_new = random.randint(0, p - 1)
        else:
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            r_new = random.randint(0, p - 1)
            
        r_old = residues[i]
        
        if r_new == r_old:
            continue
            
        # Update counts
        counts[r_old:L_MAX+1:p] -= 1
        counts[r_new:L_MAX+1:p] += 1
        
        new_L = get_L()
        
        if new_L > current_L:
            current_L = new_L
            residues[i] = r_new
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
        else:
            delta = new_L - current_L
            # Avoid overflow in exp
            if delta > -1000 * T: # Rough check
                prob = np.exp(delta / T)
            else:
                prob = 0.0
            if random.random() < prob:
                current_L = new_L
                residues[i] = r_new
            else:
                counts[r_new:L_MAX+1:p] -= 1
                counts[r_old:L_MAX+1:p] += 1
                
        T *= cooling
        if T < T_min:
            T = T_min

    return best_residues.tolist()
