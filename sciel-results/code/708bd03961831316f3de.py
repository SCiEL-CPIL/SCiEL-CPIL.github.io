import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Compute small_factor[x] = product of p^v_p(x) for all primes p < 24
    # Using vectorized numpy approach: for each prime p, for each power p^k,
    # multiply small_factor[multiple] by p for all multiples of p^k
    small_factor = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        power = p
        while power <= max_val:
            # All multiples of power get an additional factor of p
            indices = np.arange(power, max_val + 1, power)
            small_factor[indices] *= p
            power *= p
    
    # Convert to list for faster Python-level sliding window
    sf = small_factor.tolist()
    
    # Sliding window scan: for each n in [1000, 1000000],
    # window is [n+1, n+24], count distinct small_factor values
    freq = {}
    active_count = 0
    
    # Initialize for n=1000: window is [1001, 1024]
    for i in range(1, 25):
        val = sf[1000 + i]
        if val not in freq:
            freq[val] = 0
            active_count += 1
        freq[val] += 1
        
    best_score = active_count
    best_n = 1000
    
    # Slide window from n=1001 to n=1000000
    for n in range(1001, max_n + 1):
        # Time check every ~10000 iterations to reduce overhead
        if (n - 1001) % 10000 == 0:
            if time.time() > deadline:
                break
                
        # Remove sf[n] (leaving the window)
        old_val = sf[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            active_count -= 1
            del freq[old_val]
            
        # Add sf[n+24] (entering the window)
        new_val = sf[n + 24]
        if new_val not in freq:
            freq[new_val] = 0
            active_count += 1
        freq[new_val] += 1
        
        if active_count < best_score:
            best_score = active_count
            best_n = n
            if best_score == 1:
                break
                
    return best_n
