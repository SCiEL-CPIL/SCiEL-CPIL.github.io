import time
import numpy as np

def construct():
    deadline = time.time() + 82.0  # safety margin before 90s kill
    
    N = 1000000
    limit = N + 1  # need lpf for n+1 where n can be N
    
    # Sieve for smallest prime factor
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            step = i
            start = i * i
            for j in range(start, limit + 1, step):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute largest prime factor for each number
    lpf_arr = [1] * (limit + 1)
    for i in range(2, limit + 1):
        x = i
        max_pf = 1
        while x > 1:
            p = spf[x]
            max_pf = p
            while x % p == 0:
                x //= p
        lpf_arr[i] = max_pf
    
    # Convert to numpy array for fast vectorized operations
    lpf_np = np.array(lpf_arr[10000:limit + 1], dtype=np.int32)
    # lpf_np[i] corresponds to lpf_arr[10000 + i]
    # We need max(lpf[n], lpf[n+1]) for n in [10000, 1000000]
    # lpf[n] is lpf_np[n - 10000], lpf[n+1] is lpf_np[n - 10000 + 1]
    
    # Compute val[i] = max(lpf_np[i], lpf_np[i+1]) for i in [0, len-2]
    # This corresponds to n = 10000 + i
    val = np.maximum(lpf_np[:-1], lpf_np[1:])
    
    # Find the minimum value
    best_val = int(val.min())
    best_idx = int(val.argmin())
    best_n = 10000 + best_idx
    
    return best_n
