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

def _try_reduce_rank(n, m, p):
    """
    Try to find a rank-(n*m*p - 1) decomposition by modifying the schoolbook.
    Strategy: Try to merge two rank-1 terms into one by finding a linear combination.
    """
    if time.time() > _DEADLINE - 30:
        return None
    
    target_rank = n * m * p - 1  # 49
    base = _schoolbook(n, m, p)
    
    # Try to find two terms that can be merged
    # For each pair of terms (a, b), try to find a linear combination that
    # still covers all the tensor entries they covered.
    
    # A simpler approach: try to find a term c that can be eliminated by
    # adjusting other terms.
    
    # Use numerical optimization to find a rank-49 solution
    try:
        import scipy.optimize as opt
        
        best_error = float('inf')
        best_params = None
        
        # Try multiple random initializations
        max_restarts = 10
        
        for restart in range(max_restarts):
            if time.time() > _DEADLINE - 10:
                break
            
            # Initialize near the schoolbook but with one term removed
            # and the remaining terms adjusted
            np.random.seed(100 + restart)
            
            # Start with schoolbook and randomly perturb
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Copy first target_rank terms from schoolbook
            for c in range(target_rank):
                i = c // (m * p)
                j = (c // p) % m
                k = c % p
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
            
            # Add small random perturbations
            U += np.random.randn(n * m, target_rank) * 0.1
            V += np.random.randn(m * p, target_rank) * 0.1
            W += np.random.randn(p * n, target_rank) * 0.1
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
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
                # Use L-BFGS-B for better convergence
                res = opt.minimize(objective, params, method='L-BFGS-B', 
                                   options={'maxiter': 500, 'ftol': 1e-8})
                
                if res.fun < best_error:
                    best_error = res.fun
                    best_params = res.x
                    
            except Exception:
                pass
        
        if best_error < 1e-6 and best_params is not None:
            # Round to rationals and verify
            U_flat, V_flat, W_flat = np.split(best_params, [n*m*target_rank, 2*n*m*target_rank])
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            # Try rounding to nearest half-integer
            for scale in [2, 4, 8]:
                U_r = np.round(U * scale) / scale
                V_r = np.round(V * scale) / scale
                W_r = np.round(W * scale) / scale
                
                if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
            # Try integer rounding
            U_r = np.round(U)
            V_r = np.round(V)
            W_r = np.round(W)
            if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
    except Exception:
        pass
        
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find rank 49
    result = _try_reduce_rank(n, m, p)
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
