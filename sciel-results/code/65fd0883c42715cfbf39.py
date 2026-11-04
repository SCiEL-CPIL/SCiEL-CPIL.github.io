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
    """
    if time.time() > _DEADLINE - 10:
        return None
    
    target_rank = 49
    
    # Start from schoolbook decomposition
    schoolbook = _schoolbook(n, m, p)
    U_sb = np.array(schoolbook['U'], dtype=float)
    V_sb = np.array(schoolbook['V'], dtype=float)
    W_sb = np.array(schoolbook['W'], dtype=float)
    
    best_error = float('inf')
    best_params = None
    
    # We will try several optimization runs with different initializations
    
    # Strategy 1: Remove one column from schoolbook and optimize
    # Strategy 2: Random initialization
    
    np.random.seed(123)
    
    for run in range(20):
        if time.time() > _DEADLINE - 5:
            break
        
        if run < 10:
            # Remove column 'run' from schoolbook
            U_init = np.zeros((n*m, target_rank))
            V_init = np.zeros((m*p, target_rank))
            W_init = np.zeros((p*n, target_rank))
            
            new_col = 0
            for old_col in range(50):
                if old_col == run:
                    continue
                U_init[:, new_col] = U_sb[:, old_col]
                V_init[:, new_col] = V_sb[:, old_col]
                W_init[:, new_col] = W_sb[:, old_col]
                new_col += 1
            
            # Add small noise to help escape local minima
            U_init += np.random.randn(n*m, target_rank) * 0.01
            V_init += np.random.randn(m*p, target_rank) * 0.01
            W_init += np.random.randn(p*n, target_rank) * 0.01
        else:
            # Random initialization
            U_init = np.random.randn(n*m, target_rank) * 0.1
            V_init = np.random.randn(m*p, target_rank) * 0.1
            W_init = np.random.randn(p*n, target_rank) * 0.1
        
        params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(params):
            U_flat, V_flat, W_flat = np.split(params, [n*m*target_rank, 2*n*m*target_rank])
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        error += (s - 1.0)**2
            return error
        
        try:
            import scipy.optimize as opt
            # Use L-BFGS-B with more iterations
            res = opt.minimize(objective, params, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
                
        except Exception:
            pass
        
    if best_error < 1e-4 and best_params is not None:
        # Round to rationals
        U_flat, V_flat, W_flat = np.split(best_params, [n*m*target_rank, 2*n*m*target_rank])
        U = U_flat.reshape(n*m, target_rank)
        V = V_flat.reshape(m*p, target_rank)
        W = W_flat.reshape(p*n, target_rank)
        
        # Try multiple rounding schemes
        for denom in [1, 2, 4, 8, 16, 32]:
            U_r = np.round(U * denom) / denom
            V_r = np.round(V * denom) / denom
            W_r = np.round(W * denom) / denom
            
            if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
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
