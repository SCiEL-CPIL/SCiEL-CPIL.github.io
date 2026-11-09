"""Tensor decomposition search for 2x5 x 5x5 matrix multiplication."""
import numpy as np
import random
import time

DIMENSIONS = [2, 5, 5]

def construct():
    """Search for rank-minimal tensor decomposition."""
    deadline = time.time() + 140  # 140s budget, leaving 10s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build schoolbook decomposition
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    def verify(U_mat, V_mat, W_mat):
        """Check if decomposition is exact."""
        n, m, p = DIMENSIONS
        nm = n * m
        mp = m * p
        pn = p * n
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = rank_schoolbook

    def solve_for_rank(target_rank, max_restarts=20):
        nm, mp, pn = n * m, m * p, p * n
        num_vars = (nm + mp + pn) * target_rank
        
        def residuals(params):
            U_flat = params[:nm * target_rank]
            V_flat = params[nm * target_rank : nm * target_rank + mp * target_rank]
            W_flat = params[nm * target_rank + mp * target_rank :]
            
            U = U_flat.reshape(nm, target_rank)
            V = V_flat.reshape(mp, target_rank)
            W = W_flat.reshape(pn, target_rank)
            
            T = np.zeros((n, m, p))
            for r in range(target_rank):
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
                            
            return (T.flatten() - 1.0)

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None, None, None, target_rank
            
            params0 = np.random.randn(num_vars) * 0.5
            
            try:
                res = least_squares(residuals, params0, method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if res.cost < 1e-12:
                U_sol = res.x[:nm * target_rank].reshape(nm, target_rank)
                V_sol = res.x[nm * target_rank : nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                W_sol = res.x[nm * target_rank + mp * target_rank :].reshape(pn, target_rank)
                
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded):
                    return U_rounded, V_rounded, W_rounded, target_rank
                
        return None, None, None, target_rank

    for target_rank in range(20, 51):
        if time.time() > deadline - 10:
            break
            
        U_new, V_new, W_new, new_rank = solve_for_rank(target_rank, max_restarts=10)
        if U_new is not None and new_rank < best_rank:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = new_rank
            if best_rank == 20:
                break

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
