import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 10
    rank = 10
    # Total variables: U (4*10) + V (6*10) + W (6*10) = 40 + 60 + 60 = 160
    # But we can fix some to reduce symmetry.
    # Let's just solve for all.
    
    def residual(params):
        U = params[:4*rank].reshape(4, rank)
        V = params[4*rank:4*rank+6*rank].reshape(6, rank)
        W = params[4*rank+6*rank:].reshape(6, rank)
        
        # Compute sum_r U[i,j,r] V[j,k,r]
        # U indices: i*m+j
        # V indices: j*p+k
        # W indices: k*n+i
        
        # Residual for each (i, k)
        res = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    w_val = W[k*n + i, r]
                    uv_val = 0.0
                    for j in range(m):
                        uv_val += U[i*m + j, r] * V[j*p + k, r]
                    val += uv_val - w_val
                res[i, k] = val
        return res.flatten()

    # Initial guess
    # Try to initialize with small integers
    np.random.seed(42)
    best_params = None
    best_res = np.inf
    
    start_time = time.time()
    deadline = start_time + 100 # 100s limit
    
    # Try a few random initializations
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Random small integers
        params = np.random.randint(-2, 3, size=16*rank)
        
        # Run optimizer
        try:
            result = least_squares(residual, params, method='lm', max_nfev=1000)
            res_val = result.cost
            if res_val < best_res:
                best_res = res_val
                best_params = result.x
        except Exception:
            pass
        
        if best_res < 1e-6:
            break

    if best_params is not None and best_res < 1e-6:
        U = best_params[:4*rank].reshape(4, rank)
        V = best_params[4*rank:4*rank+6*rank].reshape(6, rank)
        W = best_params[4*rank+6*rank:].reshape(6, rank)
        
        # Snap to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify
        res = residual(np.concatenate([U.flatten(), V.flatten(), W.flatten()]))
        if np.max(np.abs(res)) < 1e-9:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback to rank 11 or 12 if rank 10 not found
    # Let's try rank 11
    rank = 11
    # ... similar logic ...
    
    # Fallback to schoolbook rank 12
    rank = 12
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
