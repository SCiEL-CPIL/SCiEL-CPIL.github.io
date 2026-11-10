import numpy as np
import time

def construct():
    N = 10**7
    start_time = time.time()
    
    # Compute sigma(n) for all n <= N
    # Using NumPy vectorized addition
    sigma = np.ones(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Initial best pair from seed
    best_a = 11
    best_b = 6
    best_err = abs(11/6 - 1.5) # 1/3
    
    # Search for better pairs
    # Iterate delta = 2a - 3b
    # We want small |delta| and large b
    # Error = |delta| / (2b)
    for delta in range(1, 10000):
        # Max b such that a = (3b + delta)/2 <= N
        # 3b + delta <= 2N => b <= (2N - delta)/3
        b_max = min(N, (2*N - delta) // 3)
        if b_max < 1:
            break
            
        # Theoretical minimum error for this delta is delta / (2 * b_max)
        # If this is worse than current best, no need to check this delta or larger
        min_err = delta / (2.0 * b_max)
        if min_err >= best_err:
            break
            
        # Determine starting b based on parity
        # 2a = 3b + delta
        # If delta is even, 3b must be even => b must be even
        # If delta is odd, 3b must be odd => b must be odd
        if delta % 2 == 0:
            # b must be even
            b_start = b_max if b_max % 2 == 0 else b_max - 1
        else:
            # b must be odd
            b_start = b_max if b_max % 2 == 1 else b_max - 1
            
        if b_start < 1:
            continue
            
        for b in range(b_start, 0, -1):
            # Check if current error is worse than best
            curr_err = delta / (2.0 * b)
            if curr_err >= best_err:
                break
                
            a = (3 * b + delta) // 2
            if a > N:
                continue
            if a < 1:
                break
                
            if sigma[a] == sigma[b]:
                best_err = curr_err
                best_a = a
                best_b = b
                
        # Time check
        if time.time() - start_time > 50:
            break
            
    return [int(best_a), int(best_b)]
