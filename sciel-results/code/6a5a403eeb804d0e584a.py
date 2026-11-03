import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Rank 21 decomposition: one term per (j,k) pair
    # U[i*m+j, r(j,k)] = 1 for all i (row i*m+j of U has 1 at column r(j,k))
    # V[j*p+k, r(j,k)] = 1
    # W[k*n+i, r(j,k)] = 1 for all i
    # This gives: for each (i,j,k), only term (j,k) contributes, value = 1*1*1 = 1
    
    R = m * p  # 21
    
    U_np = np.zeros((n * m, R))
    V_np = np.zeros((m * p, R))
    W_np = np.zeros((p * n, R))
    
    r = 0
    for j in range(m):
        for k in range(p):
            # U: all i values for this j
            for i in range(n):
                U_np[i * m + j, r] = 1
            # V: this (j,k)
            V_np[j * p + k, r] = 1
            # W: all i values for this k
            for i in range(n):
                W_np[k * n + i, r] = 1
            r += 1
    
    U = U_np.tolist()
    V = V_np.tolist()
    W = W_np.tolist()
    
    # Verify correctness with random test
    if time.time() < deadline - 10:
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0
                for j in range(m):
                    # Sum over all rank terms
                    for r_idx in range(R):
                        u_val = U[i * m + j][r_idx]
                        v_val = V[j * p + k][r_idx]
                        w_val = W[k * n + i][r_idx]
                        total += u_val * v_val * w_val
                C_computed[i, k] = total
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to trivial rank 42
            R2 = n * m * p
            U2 = np.zeros((n * m, R2))
            V2 = np.zeros((m * p, R2))
            W2 = np.zeros((p * n, R2))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r2 = i * m * p + j * p + k
                        U2[i * m + j, r2] = 1
                        V2[j * p + k, r2] = 1
                        W2[k * n + i, r2] = 1
            return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    return {"U": U, "V": V, "W": W}
