import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 1000000
    max_val = N + 1
    lpf = [1] * (max_val + 1)
    
    # Sieve for Largest Prime Factor
    for i in range(2, max_val + 1):
        if lpf[i] == 1:
            for j in range(i, max_val + 1, i):
                lpf[j] = i
                
    if time.time() > deadline:
        return 10000

    # Convert to numpy array for fast vectorized max
    lpf_arr = np.array(lpf, dtype=np.int32)
    
    # Calculate max(lpf[n], lpf[n+1]) for n in [10000, N-1]
    # lpf[n] is lpf_arr[10000:N]
    # lpf[n+1] is lpf_arr[10001:N+1]
    scores = np.maximum(lpf_arr[10000:N], lpf_arr[10001:N+1])
    
    # Find index of minimum score
    min_idx = np.argmin(scores)
    best_n = 10000 + min_idx
    
    return best_n
