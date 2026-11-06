import time
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0  # 5s margin
    
    N_MAX = 1000000
    N_PLUS_1 = N_MAX + 1
    SIZE = N_PLUS_1 + 1
    
    # Initialize LPF array
    # Using int32 to save memory and potentially improve cache performance
    lpf = np.zeros(SIZE, dtype=np.int32)
    
    # Optimization: Handle 2 separately
    lpf[2::2] = 2
    
    # Sieve for odd numbers
    # Check time periodically to avoid overhead in every iteration
    check_interval = 1000
    count = 0
    
    for i in range(3, SIZE, 2):
        if count % check_interval == 0:
            if time.time() - start_time > TIME_LIMIT:
                return 10000
            count += 1
        
        if lpf[i] == 0:
            # i is prime
            lpf[i::i] = i

    # Compute scores for n in [10000, 1000000]
    # n corresponds to lpf[n]
    # n+1 corresponds to lpf[n+1]
    
    # Slices
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    # Max LPF for the pair
    scores = np.maximum(lpf_n, lpf_np1)
    
    # Find the index with the minimum score
    min_idx = np.argmin(scores)
    
    best_n = 10000 + min_idx
    
    return int(best_n)
