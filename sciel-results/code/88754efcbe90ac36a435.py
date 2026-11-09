import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 5]

def _build_target(n, m, p):
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    return target

def _schoolbook(n, m, p):
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    R = n * m * p
    U = np.zeros((dim_u, R))
    V = np.zeros((dim_v, R))
    W = np.zeros((dim_w, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    return U, V, W

def _try_rank(target, R, n, m, p, time_limit, seed_base=0):
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    total_dim = dim_u + dim_v + dim_w
    
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    sb_R = n * m * p
    
    # Try multiple restarts with different initializations
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > time_limit:
            break
        
        np.random.seed(seed_base + restart * 137)
        
        # Different initialization strategies
        if restart % 3 == 0:
            # Random initialization
            U_init = np.random.randn(dim_u, R) * 0.5
            V_init = np.random.randn(dim_v, R) * 0.5
            W_init = np.random.randn(dim_w, R) * 0.5
        elif restart % 3 == 1:
            # Structured: pick R random schoolbook terms + perturbation
            indices = np.random.choice(sb_R, R, replace=False)
            U_init = U_sb[:, indices].copy()
            V_init = V_sb[:, indices].copy()
            W_init = W_sb[:, indices].copy()
            U_init += np.random.randn(dim_u, R) * 0.1
            V_init += np.random.randn(dim_v, R) * 0.1
            W_init += np.random.randn(dim_w, R) * 0.1
        else:
            # Alternative: use first R schoolbook terms
            U_init = U_sb[:, :R].copy()
            V_init = V_sb[:, :R].copy()
            W_init = W_sb[:, :R].copy()
            U_init += np.random.randn(dim_u, R) * 0.05
            V_init += np.random.randn(dim_v, R) * 0.05
            W_init += np.random.randn(dim_w, R) * 0.05
        
        def res_func(params):
            U = params[:dim_u*R].reshape(dim_u, R)
            V = params[dim_u*R:dim_u*R + dim_v*R].reshape(dim_v, R)
            W = params[dim_u*R + dim_v*R:].reshape(dim_w, R)
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            return (T_recon - target).flatten()
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(
                res_func,
                params_init,
                method='lm',
                max_nfev=200,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12
            )
            
            if result.cost < 1e-8:
                params = result.x
                U = params[:dim_u*R].reshape(dim_u, R)
                V = params[dim_u*R:dim_u*R + dim_v*R].reshape(dim_v, R)
                W = params[dim_u*R + dim_v*R:].reshape(dim_w, R)
                
                # Try to snap to nice values
                for mult in [4, 2]:
                    U_snap = np.round(U * mult) / mult
                    V_snap = np.round(V * mult) / mult
                    W_snap = np.round(W * mult) / mult
                    T_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                    if np.allclose(T_snap, target, atol=1e-9):
                        return U_snap, V_snap, W_snap
                
                # If no clean snap, return the raw solution
                T_check = np.einsum('ir,jr,kr->ijk', U, V, W)
                if np.allclose(T_check, target, atol=1e-8):
                    return U, V, W
                    
        except Exception:
            continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    target = _build_target(n, m, p)
    sb_R = n * m * p  # 45
    
    # Start with schoolbook solution
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = sb_R
    
    start_time = time.time()
    time_limit = 110  # Safety margin before 120s limit
    
    # Try to reduce rank from sb_R - 1 downward
    # Focus on the range where improvements are most likely
    for R in range(sb_R - 1, 25, -1):
        if time.time() - start_time > time_limit:
            break
        
        result = _try_rank(target, R, n, m, p, time_limit, seed_base=R * 1000)
        
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_rank = R
            best_U = U_sol
            best_V = V_sol
            best_W = W_sol
            # Don't break - keep trying for even lower rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
