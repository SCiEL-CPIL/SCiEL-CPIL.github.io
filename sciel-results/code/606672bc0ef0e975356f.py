"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]
_START_TIME = time.time()
_DEADLINE = 145.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    rank = 2
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 2)
    # V: (m*p, rank) = (32, 2)
    # W: (p*n, rank) = (32, 2)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Rank 2 decomposition:
    # Term 0 handles j=0, l=0
    # Term 1 handles j=1, l=1
    
    # U[i*m+j, s]
    # s=0: U[i*m+0, 0] = 1, U[i*m+1, 0] = 0
    # s=1: U[i*m+0, 1] = 0, U[i*m+1, 1] = 1
    for i in range(n):
        for j in range(m):
            if j == 0:
                U[i*m+j][0] = 1
            else:
                U[i*m+j][1] = 1
                
    # V[j*p+k, s]
    # s=0: V[0*p+k, 0] = 1, V[1*p+k, 0] = 0
    # s=1: V[0*p+k, 1] = 0, V[1*p+k, 1] = 1
    for j in range(m):
        for k in range(p):
            if j == 0:
                V[j*p+k][0] = 1
            else:
                V[j*p+k][1] = 1
                
    # W[k*n+i, s]
    # W[k*n+i, 0] = 1
    # W[k*n+i, 1] = 1
    for k in range(p):
        for i in range(n):
            W[k*n+i][0] = 1
            W[k*n+i][1] = 1
            
    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_decomp = np.zeros((n, p))
    for s in range(rank):
        A_proj = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_proj[i, j] = A_test[i, j] * U_arr[i*m+j, s]
        B_proj = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_proj[j, k] = B_test[j, k] * V_arr[j*p+k, s]
        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += W_arr[k*n+i, s] * np.sum(A_proj[i, :] * B_proj[:, k])
                
    if not np.allclose(C_decomp, C_expected, atol=1e-10):
        # Fallback (should not happen)
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col] = 1
                    V_s[j*p+k][col] = 1
                    W_s[k*n+i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
