import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Bisection for M
    lo, hi = 0.0, 1.0
    m = 99
    
    def check_M(M):
        h = 1.0
        for i in range(m):
            h = 1.0 - M / h
            if h < 0:
                h = 0.0
        return h <= M

    # Check if M=0.25 works (it should)
    # Bisection
    for _ in range(50):
        if time.time() > deadline:
            break
        mid = (lo + hi) / 2.0
        if check_M(mid):
            hi = mid
        else:
            lo = mid
            
    M_opt = hi
    
    # Construct sequence
    h_seq = np.zeros(N)
    h_seq[0] = 1.0
    h = 1.0
    for i in range(1, 99):
        h = 1.0 - M_opt / h
        if h < 0:
            h = 0.0
        h_seq[i] = h
    h_seq[99] = 0.0
    
    # Correct sum
    current_sum = np.sum(h_seq)
    diff = 50.0 - current_sum
    
    # Add diff to h_0
    h_seq[0] += diff
    
    # Ensure bounds
    h_seq = np.clip(h_seq, 0.0, 1.0)
    
    # Re-correct sum if clipping changed it
    current_sum = np.sum(h_seq)
    diff = 50.0 - current_sum
    h_seq[0] += diff
    
    return h_seq
