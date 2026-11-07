"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

MAX_N = 10000000

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    limit = MAX_N
    sigma_arr = np.zeros(limit + 1, dtype=np.int64)
    
    for d in range(1, limit + 1):
        sigma_arr[d::d] += d
        if d % 2000000 == 0 and time.time() > deadline - 20:
            break
    
    if time.time() > deadline - 20:
        return [11, 6]
    
    best_error = float('inf')
    best_pair = [11, 6]
    
    for b in range(2, limit + 1):
        if time.time() > deadline - 5:
            break
        sb = int(sigma_arr[b])
        a_target = (3 * b) // 2
        for da in range(-50, 51):
            a = a_target + da
            if a < 1 or a > limit:
                continue
            if a == b:
                continue
            if int(sigma_arr[a]) == sb:
                err_num = abs(2 * a - 3 * b)
                if err_num == 0:
                    continue
                err = err_num / (2 * b)
                if err < best_error:
                    best_error = err
                    best_pair = [int(a), int(b)]
    
    return [int(best_pair[0]), int(best_pair[1])]

# EVOLVE-BLOCK-END
