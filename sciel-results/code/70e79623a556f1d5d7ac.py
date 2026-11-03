"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

_START_TIME = time.time()
_DEADLINE = 145

def _build_target_tensor(n, m, p):
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    T = _build_target_tensor(n, m, p)
    Tc = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(Tc - T)) < tol

def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank_R(n, m, p, R, seed):
    if time.time() > _DEADLINE - 5:
        return None
        
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    
    rng = np.random.RandomState(seed)
    # Random initialization
    x0 = rng.randn(R * (nm + mp + pn)) * 0.1
    
    def residual(x):
        U = x[:R*nm].reshape(nm, R)
        V = x[R*nm:R*nm+R*mp].reshape(mp, R)
        W = x[R*nm+R*mp:].reshape(pn, R)
        Tc = np.einsum('ar,br,cr->abc', U, V, W)
        return (Tc - T).flatten()
    
    try:
        # Use Levenberg-Marquardt for small problems
        res = least_squares(residual, x0, method='lm', max_nfev=500)
    except Exception:
        return None
        
    if res.cost < 1e-10:
        U = res.x[:R*nm].reshape(nm, R)
        V = res.x[R*nm:R*nm+R*mp].reshape(mp, R)
        W = res.x[R*nm+R*mp:].reshape(pn, R)
        
        # Try to round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-9):
            return U_r, V_r, W_r
        
        # If rounding fails, check if raw is valid (unlikely to be integer/half-integer but maybe)
        # The task requires integer/half-integer entries.
        # If raw is not valid as half-integers, we cannot return it.
        # So we must ensure the rounded version is valid.
        pass

    return None

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Schoolbook
    U24, V24, W24 = _construct_schoolbook(n, m, p)
    
    # 2. Try rank 23
    seeds = [42, 123, 456, 789, 1024, 2048, 3000, 4000, 5000, 6000, 7000, 8000]
    for seed in seeds:
        if time.time() > _DEADLINE - 5:
            break
        result = _try_rank_R(n, m, p, 23, seed)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # 3. Fallback
    return {
        "U": U24.tolist(),
        "V": V24.tolist(),
        "W": W24.tolist()
    }
# EVOLVE-BLOCK-END
