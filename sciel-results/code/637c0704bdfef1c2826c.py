import numpy as np
import time
import random

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # 1. Establish the baseline: Schoolbook Rank 27
    best_rank = 27
    best_U = np.zeros((n * m, 27), dtype=float)
    best_V = np.zeros((m * p, 27), dtype=float)
    best_W = np.zeros((p * n, 27), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1

    # 2. Define the target tensor T
    T_target = np.zeros((n * m, m * p, p * n))
    eq_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                eq_indices.append((a, b, c))
                
    num_eqs = len(eq_indices)
    
    def verify_decomposition(U, V, W, tol=1e-9):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(T_recon, T_target, atol=tol):
            return False
        return True

    def check_half_integer(mat, tol=1e-6):
        doubled = mat * 2
        rounded = np.round(doubled)
        if np.max(np.abs(doubled - rounded)) < tol:
            return rounded / 2.0
        return None

    # 3. Search for lower rank decompositions using structured random initialization
    for target_rank in [23, 24, 25, 26]:
        if time.time() > deadline - 15:
            break
            
        max_trials = 100 if target_rank == 23 else 200
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
                
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            
            for c in range(target_rank):
                for idx in range(n * m):
                    if random.random() < 0.35:
                        U[idx, c] = random.choice([-1, 1])
                for idx in range(m * p):
                    if random.random() < 0.35:
                        V[idx, c] = random.choice([-1, 1])
                        
            M = np.zeros((num_eqs, p * n * target_rank))
            
            for e_idx, (a, b, c) in enumerate(eq_indices):
                for r in range(target_rank):
                    coeff = U[a, r] * V[b, r]
                    if coeff != 0:
                        M[e_idx, c * target_rank + r] = coeff
                        
            try:
                W_vec, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(num_eqs), rcond=None)
                
                if residuals and residuals[0] > 1e-8:
                    continue
                    
                W_sol = W_vec.reshape(target_rank, p * n)
                
                W_snapped = check_half_integer(W_sol)
                if W_snapped is None:
                    continue
                
                if verify_decomposition(U, V, W_snapped, tol=1e-9):
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W_snapped
                    break
            except Exception:
                continue
        
        if best_rank == target_rank:
            break
            
    def to_list(mat):
        if isinstance(mat, np.ndarray):
            return mat.tolist()
        return mat

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }

def construct_seed():
    n, m, p = 3, 3, 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
