"""Combined: Rank-18 base with robust Rank-17 search using Non-linear Least Squares."""
import numpy as np
import time
import sys
from itertools import product
import scipy.optimize as opt

DIMENSIONS = [2, 2, 5]

def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T

def _check_decomposition(U, V, W, T):
    """Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w."""
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    result = np.zeros_like(T, dtype=np.float64)
    for r in range(U_a.shape[1]):
        result += np.einsum('u,v,w->uvw', U_a[:,r], V_a[:,r], W_a[:,r])
    return np.allclose(result, T.astype(np.float64), atol=1e-9)

def _build_rank18(n, m, p):
    """Build the known rank-18 Strassen+schoolbook decomposition."""
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return U, V, W

def _try_rank17_nls(n, m, p, T, deadline):
    """Try to find a rank-17 decomposition using Non-Linear Least Squares."""
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    T_flat = T.flatten()
    
    def residual(params):
        # params: flattened U, V, W
        U_flat, V_flat, W_flat = params[:4*17], params[4*17:4*17+10*17], params[4*17+10*17:]
        U = U_flat.reshape(4, 17)
        V = V_flat.reshape(10, 17)
        W = W_flat.reshape(10, 17)
        
        # Compute tensor
        result = np.zeros((4, 10, 10), dtype=np.float64)
        for r in range(17):
            result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
        
        return (result.flatten() - T_flat)

    # Strategy: Remove each term from Rank 18 and optimize
    for term_to_remove in range(18):
        if time.time() > deadline:
            break
            
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].flatten()
        V_init = V18_a[:, idx].flatten()
        W_init = W18_a[:, idx].flatten()
        
        x0 = np.concatenate([U_init, V_init, W_init])
        
        try:
            sol = opt.least_squares(residual, x0, method='trf', max_nfev=2000, xtol=1e-15, ftol=1e-15)
            
            if sol.success or sol.cost < 1e-10:
                U_sol = sol.x[:4*17].reshape(4, 17)
                V_sol = sol.x[4*17:4*17+10*17].reshape(10, 17)
                W_sol = sol.x[4*17+10*17:].reshape(10, 17)
                
                # Check exactness
                if _check_decomposition(U_sol, V_sol, W_sol, T):
                    # Try to snap to half-integers
                    for denom in [2, 4, 8]:
                        U_s = np.round(U_sol * denom) / denom
                        V_s = np.round(V_sol * denom) / denom
                        W_s = np.round(W_sol * denom) / denom
                        if _check_decomposition(U_s, V_s, W_sol, T): # Note: W_sol might be better, check all
                             if _check_decomposition(U_s, V_s, W_s, T):
                                return U_s.tolist(), V_s.tolist(), W_s.tolist()
                    return U_sol.tolist(), V_sol.tolist(), W_sol.tolist()
        except Exception:
            pass

    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    T = _build_target()
    
    # First, build the known rank-18 decomposition
    U18, V18, W18 = _build_rank18(n, m, p)
    
    # Verify rank-18
    if _check_decomposition(U18, V18, W18, T):
        # Try to find rank-17
        result17 = _try_rank17_nls(n, m, p, T, deadline)
        if result17 is not None:
            U, V, W = result17
            if _check_decomposition(U, V, W, T):
                return {"U": U, "V": V, "W": W}
    
    # Fallback: return rank-18
    return {
        "U": U18,
        "V": V18,
        "W": W18
    }
