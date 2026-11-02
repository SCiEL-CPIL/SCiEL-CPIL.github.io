"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # The matrix multiplication tensor T_{(i,j),(j,k),(k,i)} = 1
    # can be decomposed as sum over i of delta_{i,i'} * 1 * delta_{i,i'}
    # This gives rank n = 3 decomposition.
    
    rank = n
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(n):
        # U: index (i*m + j, r) = 1 if i == r
        for i in range(n):
            for j in range(m):
                if i == r:
                    U[i * m + j][r] = 1
        
        # V: index (j*p + k, r) = 1 for all j, k
        for j in range(m):
            for k in range(p):
                V[j * p + k][r] = 1
        
        # W: index (k*n + i, r) = 1 if i == r
        for k in range(p):
            for i in range(n):
                if i == r:
                    W[k * n + i][r] = 1
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1
    
    T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    
    if not np.allclose(T_decomp, target, atol=1e-10):
        # Fallback to schoolbook if verification fails (shouldn't happen)
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][c] = 1
                    V_s[j*p+k][c] = 1
                    W_s[k*n+i][c] = 1
                    c += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
