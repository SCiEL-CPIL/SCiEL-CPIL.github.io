import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()
_HARD_DEADLINE = 115

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + _HARD_DEADLINE

    # Get the rank-14 solution first as a fallback
    U14, V14, W14 = _get_rank14_solution(n, m, p)
    
    # Verify it
    err14 = _verify_decomposition(U14, V14, W14, n, m, p)
    
    if err14 < 1e-9:
        # Try to find lower rank
        for target_rank in [13, 12]:
            if time.time() > deadline - 15:
                break
            U, V, W = _search_lower_rank(n, m, p, target_rank, deadline)
            if U is not None:
                err = _verify_decomposition(U, V, W, n, m, p)
                if err < 1e-9:
                    return _to_dict(U, V, W)
        
        return _to_dict(U14, V14, W14)
    else:
        # Fallback to schoolbook (rank 36)
        return _construct_schoolbook(n, m, p)

def _get_rank14_solution(n, m, p):
    # Rank-14 decomposition for 2x3x6
    # This is the solution from the seed program
    U14 = np.array([
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1]
    ], dtype=np.float64)

    V14 = np.array([
        [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, -1, 1, 0, 0],
        [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, -1, 1, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, 0, 0]
    ], dtype=np.float64)

    W14 = np.array([
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ], dtype=np.float64)
    
    return U14, V14, W14

def _verify_decomposition(U, V, W, n, m, p):
    r = U.shape[1]
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    err = np.max(np.abs(T - 1.0))
    return err

def _residual(params, n, m, p):
    r = params.size // (n*m + m*p + p*n)
    U = params[0: n*m*r].reshape(n*m, r)
    V = params[n*m*r: n*m*r + m*p*r].reshape(m*p, r)
    W = params[n*m*r + m*p*r:].reshape(p*n, r)
    
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    residual = (T - 1.0).flatten()
    return np.sum(residual**2)

def _search_lower_rank(n, m, p, target_rank, deadline):
    r = target_rank
    best_loss = np.inf
    best_params = None
    
    # Number of restarts based on available time
    # Each optimization takes ~0.5-2s
    max_restarts = 30
    
    for i in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Random initialization
        np.random.seed(42 + i)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Optimize
        try:
            result = minimize(_residual, params, args=(n, m, p), method='Powell', 
                           options={'maxiter': 100, 'ftol': 1e-12, 'xtol': 1e-12})
            loss = result.fun
            if loss < best_loss:
                best_loss = loss
                best_params = result.x
            if best_loss < 1e-10:
                break
        except Exception:
            continue
            
    if best_params is not None and best_loss < 1e-9:
        U = best_params[0: n*m*r].reshape(n*m, r)
        V = best_params[n*m*r: n*m*r + m*p*r].reshape(m*p, r)
        W = best_params[n*m*r + m*p*r:].reshape(p*n, r)
        return U, V, W
    
    return None, None, None

def _construct_schoolbook(n, m, p):
    r = n * m * p
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    return _to_dict(U, V, W)

def _to_dict(U, V, W):
    def clean(arr):
        out = []
        for row in arr:
            crow = []
            for x in row:
                xi = round(float(x))
                if abs(float(x) - xi) < 1e-9:
                    crow.append(int(xi))
                else:
                    h = round(2.0 * float(x))
                    if abs(2.0 * float(x) - h) < 1e-9:
                        crow.append(h / 2.0)
                    else:
                        crow.append(float(x))
            out.append(crow)
        return out
    return {"U": clean(U), "V": clean(V), "W": clean(W)}
