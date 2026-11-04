import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best = {"U": U_school, "V": V_school, "W": W_school}

    # Target tensor T[i, j, k] = 1
    # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    
    def compute_residual(params, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape((n, m, R))
        V = V_flat.reshape((m, p, R))
        W = W_flat.reshape((p, n, R))
        
        residual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i, j, :] * V[j, k, :] * W[k, i, :])
                    residual[i, j, k] = val - 1.0
        return residual.flatten()

    def try_rank(R, num_restarts=10):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W
        
        for restart in range(num_restarts):
            np.random.seed(restart * 100 + R)
            params = np.random.randn(total_size)
            
            # Try to solve
            try:
                res = least_squares(compute_residual, params, args=(R,), method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if res.cost < 1e-10:
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U+size_V]
                W_flat = res.x[size_U+size_V:size_U+size_V+size_W]
                
                U = U_flat.reshape((n, m, R))
                V = V_flat.reshape((m, p, R))
                W = W_flat.reshape((p, n, R))
                
                # Snap to half-integers
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Verify snapped solution
                res_snapped = compute_residual(
                    np.concatenate([U_snapped.flatten(), V_snapped.flatten(), W_snapped.flatten()]), 
                    R
                )
                if np.max(np.abs(res_snapped)) < 1e-9:
                    U_list = [[U_snapped[i, j, c] for c in range(R)] for i in range(n) for j in range(m)]
                    V_list = [[V_snapped[j, k, c] for c in range(R)] for j in range(m) for k in range(p)]
                    W_list = [[W_snapped[k, i, c] for c in range(R)] for k in range(p) for i in range(n)]
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
        return None

    start_time = time.time()
    deadline = start_time + 100
    
    # Try ranks from 24 up to 35
    for R in range(24, 36):
        if time.time() > deadline:
            break
            
        result = try_rank(R, num_restarts=20)
        if result is not None:
            return result

    return best
