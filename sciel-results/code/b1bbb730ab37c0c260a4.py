"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    def check_time():
        return time.time() > deadline
    
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        if not U or not V or not W:
            return False
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        if U_arr.shape[0] != n * m or V_arr.shape[0] != m * p or W_arr.shape[0] != p * n:
            return False
        rank = U_arr.shape[1]
        if rank == 0 or V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1
        return np.allclose(T_decomp, T_target, atol=1e-9)
    
    # Schoolbook decomposition: rank = n*m*p = 54
    # Each (i,j,k) triple gets its own term
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Should not reach here, but return anyway
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
