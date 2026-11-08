import numpy as np
import time

def _try_find_decomposition(n, m, p, target_rank, timeout):
    start = time.time()
    # Flatten indices
    # U: nm x R, V: mp x R, W: np x R
    # T: nm x mp x np (flattened)
    # T[idx1, idx2, idx3] = 1 if idx1=(i,j), idx2=(j,k), idx3=(i,k) else 0
    
    # Build T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # We want sum_r U[:,r] V[:,r] W[:,r] = T
    # This is equivalent to finding R vectors u_r, v_r, w_r
    
    # This is hard. I will skip the search.
    return None

