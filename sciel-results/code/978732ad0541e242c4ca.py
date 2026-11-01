import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 2, 3, 9
    
    # Use Schoolbook decomposition
    # Rank = n * m * p = 2 * 3 * 9 = 54
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Iterate over all multiplications
    # r index corresponds to (i, j, k)
    # i in 0..n-1 (row of A, row of C)
    # j in 0..m-1 (col of A, row of B)
    # k in 0..p-1 (col of B, col of C)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: A(i, j) -> i*m + j
                u_idx = i * m + j
                U[u_idx, r] = 1.0
                
                # V index: B(j, k) -> j*p + k
                v_idx = j * p + k
                V[v_idx, r] = 1.0
                
                # W index: C(i, k) -> k*n + i (Trace convention W[k*n+i, r])
                w_idx = k * n + i
                W[w_idx, r] = 1.0
                
                r += 1
                
                if time.time() - start_time > 110:
                    # Safety margin, though this loop is fast
                    break
            
            if time.time() - start_time > 110:
                break
        
        if time.time() - start_time > 110:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
