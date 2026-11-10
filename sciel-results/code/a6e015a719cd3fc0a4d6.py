import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_n = 1000000
    max_val = max_n + 24
    
    small_factor = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        power = p
        while power <= max_val:
            indices = np.arange(power, max_val + 1, power)
            small_factor[indices] *= p
            power *= p
            
    sf_list = small_factor.tolist()
    
    # Max possible small factor is max_val (1000024)
    # But actually much smaller. 10^6 is safe.
    max_sf = max_val + 1 
    counts = [0] * (max_sf)
    
    best_score = 25
    best_n = 1000
    
    current_distinct = 0
    
    # Init window n=1000
    for i in range(1001, 1025):
        v = sf_list[i]
        if counts[v] == 0:
            current_distinct += 1
        counts[v] += 1
        
    if current_distinct < best_score:
        best_score = current_distinct
        best_n = 1000
        if best_score == 1:
            return best_n
            
    for n in range(1001, max_n + 1):
        if n % 50000 == 0:
            if time.time() > deadline:
                break
                
        # Remove outgoing sf_list[n]
        val_out = sf_list[n]
        counts[val_out] -= 1
        if counts[val_out] == 0:
            current_distinct -= 1
            
        # Add incoming sf_list[n+24]
        val_in = sf_list[n + 24]
        if counts[val_in] == 0:
            current_distinct += 1
        counts[val_in] += 1
        
        if current_distinct < best_score:
            best_score = current_distinct
            best_n = n
            if best_score == 1:
                break
                
    return best_n
