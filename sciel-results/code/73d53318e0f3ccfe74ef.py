import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy: Use numerical optimization to find a decomposition of rank R.
    We try ranks starting from a lower bound up to the known bound 44.
    """
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 8
    
    def error_func(params, R):
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        U = params[0 : R*size_U].reshape(R, size_U)
        V = params[R*size_U : R*size_U + R*size_V].reshape(R, size_V)
        W = params[R*size_U + R*size_V : R*size_U + R*size_V + R*size_W].reshape(R, size_W)
        
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    w_col = W[:, k*n + i]
                    u_row = U[:, i*m + j]
                    v_row = V[:, j*p + k]
                    
                    val = np.sum(w_col * u_row * v_row)
                    target = 1.0 if j == k else 0.0
                    residuals.append(val - target)
                    
        return np.array(residuals)

    def try_rank(R, max_restarts=5):
        size_U = n * m
        size_V = m * p
        size_W = p * n
        total_size = R * (size_U + size_V + size_W)
        
        for restart in range(max_restarts):
            if time.time() - start_time > timeout:
                return None
                
            np.random.seed(42 + restart)
            params = np.random.randn(total_size) * 0.1
            
            try:
                res = least_squares(error_func, params, args=(R,), method='lm', max_nfev=5000)
            except Exception:
                continue
                
            if res.success and res.cost < 1e-12:
                U = res.x[0 : R*size_U].reshape(R, size_U)
                V = res.x[R*size_U : R*size_U + R*size_V].reshape(R, size_V)
                W = res.x[R*size_U + R*size_V : R*size_U + R*size_V + R*size_W].reshape(R, size_W)
                
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                residuals_snapped = error_func(
                    np.concatenate([U_snapped.flatten(), V_snapped.flatten(), W_snapped.flatten()]), 
                    R
                )
                
                if np.max(np.abs(residuals_snapped)) < 1e-9:
                    U_list = U_snapped.T.tolist()
                    V_list = V_snapped.T.tolist()
                    W_list = W_snapped.T.tolist()
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    def get_fallback():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
        
        for block_idx in range(4):
            k_start = block_idx * 2
            steps = [
                {'U': {0*m + 1: 1, 1*m + 2: 1}, 'V': {1*p + k_start: 1, 2*p + (k_start+1): 1}, 'W': {k_start*n + 0: 1, (k_start+1)*n + 1: 1}},
                {'U': {1*m + 1: 1, 1*m + 2: 1}, 'V': {1*p + k_start: 1}, 'W': {k_start*n + 1: 1, (k_start+1)*n + 1: -1}},
                {'U': {0*m + 1: 1}, 'V': {1*p + (k_start+1): 1, 2*p + (k_start+1): -1}, 'W': {(k_start+1)*n + 0: 1, (k_start+1)*n + 1: 1}},
                {'U': {1*m + 2: 1}, 'V': {2*p + k_start: 1, 1*p + k_start: -1}, 'W': {k_start*n + 0: 1, k_start*n + 1: 1}},
                {'U': {0*m + 1: 1, 0*m + 2: 1}, 'V': {2*p + (k_start+1): 1}, 'W': {k_start*n + 0: -1, (k_start+1)*n + 0: 1}},
                {'U': {1*m + 1: 1, 0*m + 1: -1}, 'V': {1*p + k_start: 1, 1*p + (k_start+1): 1}, 'W': {(k_start+1)*n + 1: 1}},
                {'U': {0*m + 2: 1, 1*m + 2: -1}, 'V': {2*p + k_start: 1, 2*p + (k_start+1): 1}, 'W': {k_start*n + 0: 1}}
            ]
            for step in steps:
                for u_idx, val in step['U'].items():
                    U[u_idx][col] += val
                for v_idx, val in step['V'].items():
                    V[v_idx][col] += val
                for w_idx, val in step['W'].items():
                    W[w_idx][col] += val
                col += 1
        return {"U": U, "V": V, "W": W}

    best_result = None
    
    for R in range(40, 45):
        if time.time() - start_time > timeout:
            break
        result = try_rank(R, max_restarts=10)
        if result is not None:
            best_result = result
            break
            
    if best_result is None:
        best_result = get_fallback()
        
    return best_result
# EVOLVE-BLOCK-END
