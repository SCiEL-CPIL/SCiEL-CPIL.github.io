import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank_49(n, m, p):
    """
    Attempt to find rank-49 decomposition for 2x5x5 matrix multiplication.
    Uses numerical optimization with multiple strategies, then rounds to rationals.
    """
    if time.time() > _DEADLINE - 20:
        return None

    target_rank = 49
    # Dimensions: n=2, m=5, p=5
    # U: (10, 49), V: (25, 49), W: (10, 49)
    
    try:
        import scipy.optimize as opt
        
        best_error = float('inf')
        best_params = None
        
        # Precompute valid indices
        valid_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    valid_indices.append((u_idx, v_idx, w_idx))
        
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > _DEADLINE - 5:
                break
            
            np.random.seed(100 + restart)
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(params):
                U_flat, V_flat, W_flat = np.split(params, [n*m*target_rank, 2*n*m*target_rank])
                U = U_flat.reshape(n*m, target_rank)
                V = V_flat.reshape(m*p, target_rank)
                W = W_flat.reshape(p*n, target_rank)
                
                error = 0.0
                for u_idx, v_idx, w_idx in valid_indices:
                    s = 0.0
                    for c in range(target_rank):
                        s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    error += (s - 1.0)**2
                return error
            
            try:
                # Use Nelder-Mead for robustness, limited iterations
                res = opt.minimize(objective, params, method='Nelder-Mead', 
                                   options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
                
                if res.fun < best_error:
                    best_error = res.fun
                    best_params = res.x
                    
            except Exception:
                pass
        
        if best_error < 1e-6 and best_params is not None:
            U_flat, V_flat, W_flat = np.split(best_params, [n*m*target_rank, 2*n*m*target_rank])
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            # Try rounding to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
            # Try rounding to nearest integer
            U_r = np.round(U)
            V_r = np.round(V)
            W_r = np.round(W)
            if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # Try rounding to nearest quarter
            U_r = np.round(U * 4) / 4
            V_r = np.round(V * 4) / 4
            W_r = np.round(W * 4) / 4
            if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
    except Exception:
        pass
        
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find rank 49
    result = _try_rank_49(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
