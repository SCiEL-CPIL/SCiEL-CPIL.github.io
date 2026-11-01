import time
import numpy as np
import random

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Define the target tensor indices (i, j, k) -> (a, b, c)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
    
    target_arr = np.array(target_indices)
    
    def is_valid(U_mat, V_mat, W_mat, rank):
        """Check if decomposition is valid using vectorized operations."""
        a_idx = target_arr[:, 0]
        b_idx = target_arr[:, 1]
        c_idx = target_arr[:, 2]
        U_t = U_mat[a_idx, :rank]
        V_t = V_mat[b_idx, :rank]
        W_t = W_mat[c_idx, :rank]
        prod = U_t * V_t * W_t
        sums = np.sum(prod, axis=1)
        return np.all(np.abs(sums - 1.0) < 1e-9)
    
    # Schoolbook decomposition: rank = n*m*p = 52
    R0 = n * m * p
    U = np.zeros((nm, R0), dtype=float)
    V = np.zeros((mp, R0), dtype=float)
    W = np.zeros((pn, R0), dtype=float)
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col_idx] = 1.0
                V[j * p + k, col_idx] = 1.0
                W[k * n + i, col_idx] = 1.0
                col_idx += 1
                
    best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R0
    
    # Fast check: if we have time, try a few greedy removals
    if time.time() < deadline - 10:
        U_work = best_U.copy()
        V_work = best_V.copy()
        W_work = best_W.copy()
        current_rank = best_R
        
        # Try removing columns in a deterministic order first
        for col in range(current_rank):
            if time.time() > deadline:
                break
            if col >= current_rank:
                continue
            
            U_test = U_work.copy()
            V_test = V_work.copy()
            W_test = W_work.copy()
            U_test[:, col] = 0
            V_test[:, col] = 0
            W_test[:, col] = 0
            
            if is_valid(U_test, V_test, W_test, current_rank):
                U_work = np.delete(U_work, col, axis=1)
                V_work = np.delete(V_work, col, axis=1)
                W_work = np.delete(W_work, col, axis=1)
                current_rank -= 1
                if current_rank < best_R:
                    best_U, best_V, best_W, best_R = U_work.copy(), V_work.copy(), W_work.copy(), current_rank
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
