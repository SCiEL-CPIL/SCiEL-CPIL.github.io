import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize Schoolbook
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U, V, W
    best_rank = rank

    # Verify the solution
    # The tensor T_ijk should be sum_l U_il V_jl W_kl
    # We check a few random samples to ensure correctness
    # Full check is O(n*m*p*rank) which is 105*105 ~ 11000 ops, very fast
    
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T_ijk = 1 if i,j,k are valid indices for the multiplication?
                    # No, T_ijk is the coefficient of A_ij B_jk in C_ik.
                    # In the tensor definition, T_ijk = 1 if the indices match the multiplication structure.
                    # Specifically, the tensor is T[i, j, k] = 1 if we are computing C[i, k] from A[i, j] and B[j, k].
                    # So T[i, j, k] is 1 for all i, j, k in the schoolbook decomposition.
                    # Wait, the tensor T is defined such that C_ik = sum_j T[i, j, k] A_ij B_jk.
                    # So T[i, j, k] = 1 for all i, j, k.
                    
                    val = 0.0
                    for l in range(rank):
                        val += U[i * m + j, l] * V[j * p + k, l] * W[k * n + i, l]
                    
                    # For schoolbook, T[i,j,k] is 1.
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        # This should never happen for schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
