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
        # For each target (a,b,c), compute sum_r U[a,r]*V[b,r]*W[c,r]
        # Vectorized: for each r, compute outer products
        # Efficient: compute U_target = U_mat[a_indices, :rank] shape (52, rank)
        # V_target = V_mat[b_indices, :rank] shape (52, rank)
        # W_target = W_mat[c_indices, :rank] shape (52, rank)
        # Then sum over r of U_target[i,r]*V_target[i,r]*W_target[i,r]
        a_idx = target_arr[:, 0]
        b_idx = target_arr[:, 1]
        c_idx = target_arr[:, 2]
        U_t = U_mat[a_idx, :rank]  # (52, rank)
        V_t = V_mat[b_idx, :rank]  # (52, rank)
        W_t = W_mat[c_idx, :rank]  # (52, rank)
        # Element-wise product then sum over rank
        prod = U_t * V_t * W_t  # (52, rank)
        sums = np.sum(prod, axis=1)  # (52,)
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
    
    # Try greedy removal with random restarts
    for restart in range(10):
        if time.time() > deadline:
            break
        
        U_work = best_U.copy()
        V_work = best_V.copy()
        W_work = best_W.copy()
        current_rank = best_R
        
        columns_to_try = list(range(current_rank))
        random.shuffle(columns_to_try)
        
        for col in columns_to_try:
            if time.time() > deadline:
                break
            if col >= current_rank:
                continue
            
            # Try removing column col
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
    
    # Try local search: perturb entries to enable more removals
    if time.time() < deadline - 20:
        U_work = best_U.copy()
        V_work = best_V.copy()
        W_work = best_W.copy()
        current_rank = best_R
        
        # Try to reduce rank by 1 by solving for new columns
        # This is more complex, so we'll try random perturbations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Randomly perturb some entries
            U_pert = U_work.copy()
            V_pert = V_work.copy()
            W_pert = W_work.copy()
            
            for _ in range(3):
                r = random.randint(0, current_rank - 1)
                i = random.randint(0, nm - 1)
                j = random.randint(0, mp - 1)
                k = random.randint(0, pn - 1)
                U_pert[i, r] += random.choice([-0.5, 0.5])
                V_pert[j, r] += random.choice([-0.5, 0.5])
                W_pert[k, r] += random.choice([-0.5, 0.5])
            
            if is_valid(U_pert, V_pert, W_pert, current_rank):
                # Try to remove a column
                cols = list(range(current_rank))
                random.shuffle(cols)
                for col in cols:
                    if time.time() > deadline:
                        break
                    U_test = U_pert.copy()
                    V_test = V_pert.copy()
                    W_test = W_pert.copy()
                    U_test[:, col] = 0
                    V_test[:, col] = 0
                    W_test[:, col] = 0
                    if is_valid(U_test, V_test, W_test, current_rank):
                        U_pert = np.delete(U_pert, col, axis=1)
                        V_pert = np.delete(V_pert, col, axis=1)
                        W_pert = np.delete(W_pert, col, axis=1)
                        current_rank -= 1
                        if current_rank < best_R:
                            best_U, best_V, best_W, best_R = U_pert.copy(), V_pert.copy(), W_pert.copy(), current_rank
                        break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
