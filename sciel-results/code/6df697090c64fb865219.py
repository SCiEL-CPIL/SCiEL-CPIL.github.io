import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    N_MIN = 10000
    N_MAX = 1000000
    MAX_VAL = N_MAX + 1

    # Use NumPy for efficient sieve
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int32)
    
    for i in range(2, MAX_VAL + 1):
        if lpf[i] == 0:
            lpf[i::i] = i

    # Vectorized scan for minimum max(lpf[n], lpf[n+1])
    vals_n = lpf[N_MIN:N_MAX+1]
    vals_np1 = lpf[N_MIN+1:N_MAX+2]
    
    max_vals = np.maximum(vals_n, vals_np1)
    min_idx = np.argmin(max_vals)
    best_n = N_MIN + min_idx
    
    return int(best_n)
