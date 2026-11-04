import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _objective(params, n, m, p, r):
    """
    Objective function for optimizing U and V.
    For fixed U, V, we solve for W in the least squares sense for each c.
    The objective is the sum of squared residuals.
    """
    U_flat = params[:n * m * r]
    V_flat = params[n * m * r:]
    U = U_flat.reshape(n * m, r)
    V = V_flat.reshape(m * p, r)
    
    total_err = 0.0
    
    # c index corresponds to k*n + i
    for c in range(p * n):
        k = c // n
        i = c % n
        
        # A_c has shape (m, r)
        # A_c[j, r_idx] = U[i*m + j, r_idx] * V[j*p + k, r_idx]
        A = np.zeros((m, r))
        for j in range(m):
            u_row = U[i * m + j]
            v_row = V[j * p + k]
            A[j] = u_row * v_row
            
        # We want to solve A * W_c = ones(m)
        # W_c has shape (r,)
        # Least squares solution: W_c = A^T (A A^T)^{-1} b
        # If A A^T is singular, use pinv
        b = np.ones(m)
        
        try:
            # Use pinv for stability
            W_c = np.linalg.pinv(A) @ b
            residual = A @ W_c - b
            total_err += np.sum(residual * residual)
        except Exception:
            total_err += 1e6
            
    return total_err

def _try_rank(n, m, p, r, time_budget):
    """
    Try to find a decomposition of rank r.
    """
    start_time = time.time()
    
    n_params = (n * m + m * p) * r
    
    best_err = np.inf
    best_params = None
    
    # Use multiple restarts with varied initialization strategies
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget * 0.95:
            break
            
        # Different initialization strategies
        if restart % 3 == 0:
            # Random small values
            np.random.seed(hash((n, m, p, r, restart)) % (2**32))
            params_init = np.random.randn(n_params) * 0.5
        elif restart % 3 == 1:
            # Random values with some structure
            np.random.seed(hash((n, m, p, r, restart)) % (2**32))
            params_init = np.random.randn(n_params) * 0.3
            # Add some zeros to encourage sparsity
            if r > 10:
                params_init[np.random.choice(n_params, size=n_params//4, replace=False)] = 0
        else:
            # Random values with larger scale
            np.random.seed(hash((n, m, p, r, restart)) % (2**32))
            params_init = np.random.randn(n_params) * 0.8
        
        try:
            result = minimize(
                _objective,
                params_init,
                args=(n, m, p, r),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 300}
            )
            
            err = result.fun
            if err < best_err:
                best_err = err
                best_params = result.x.copy()
                
            if err < 1e-6:
                # Found a good solution, do a polish pass
                try:
                    result2 = minimize(
                        _objective,
                        best_params,
                        args=(n, m, p, r),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 500}
                    )
                    if result2.fun < best_err:
                        best_err = result2.fun
                        best_params = result2.x.copy()
                except Exception:
                    pass
                break
                
        except Exception:
            continue
            
    return best_err, best_params

def _extract_decomposition(params, n, m, p, r):
    """
    Extract U, V, W from optimized params.
    """
    U_flat = params[:n * m * r]
    V_flat = params[n * m * r:]
    U = U_flat.reshape(n * m, r)
    V = V_flat.reshape(m * p, r)
    
    W = np.zeros((p * n, r))
    
    for c in range(p * n):
        k = c // n
        i = c % n
        
        A = np.zeros((m, r))
        for j in range(m):
            u_row = U[i * m + j]
            v_row = V[j * p + k]
            A[j] = u_row * v_row
            
        b = np.ones(m)
        W_c = np.linalg.pinv(A) @ b
        W[c] = W_c
        
    return U, V, W

def _snap_to_rational(U, V, W, n, m, p, tol=1e-6):
    """
    Snap U, V, W to half-integers or integers.
    """
    T = _build_tensor(n, m, p)
    
    # Try half-integers first
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    recon = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
    err = np.sum((recon - T) ** 2)
    
    if err < tol:
        return U_snap, V_snap, W_snap
        
    # Try integers
    U_snap = np.round(U)
    V_snap = np.round(V)
    W_snap = np.round(W)
    
    recon = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
    err = np.sum((recon - T) ** 2)
    
    if err < tol:
        return U_snap, V_snap, W_snap
        
    # Try quarter-integers
    U_snap = np.round(U * 4) / 4
    V_snap = np.round(V * 4) / 4
    W_snap = np.round(W * 4) / 4
    
    recon = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
    err = np.sum((recon - T) ** 2)
    
    if err < tol:
        return U_snap, V_snap, W_snap
        
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard with safety margin
    total_start = time.time()
    time_budget = 115.0  # 150s limit, leave 35s safety margin
    
    # Fallback schoolbook rank 24
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Try ranks from 19 to 23
    for r in range(19, 24):
        elapsed = time.time() - total_start
        remaining = time_budget - elapsed
        if remaining < 10.0:
            break
            
        # Allocate time: give more time to lower ranks
        if r == 19:
            rank_budget = min(remaining * 0.4, 40.0)
        elif r == 20:
            rank_budget = min(remaining * 0.3, 30.0)
        elif r == 21:
            rank_budget = min(remaining * 0.2, 20.0)
        else:
            rank_budget = min(remaining * 0.1, 10.0)
            
        best_err, best_params = _try_rank(n, m, p, r, rank_budget)
        
        if best_err < 1e-6 and best_params is not None:
            U, V, W = _extract_decomposition(best_params, n, m, p, r)
            
            # Snap to rational
            snapped = _snap_to_rational(U, V, W, n, m, p)
            if snapped is not None:
                U_snap, V_snap, W_snap = snapped
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
            
            # If snapping fails, try with wider tolerance
            snapped = _snap_to_rational(U, V, W, n, m, p, tol=1e-4)
            if snapped is not None:
                U_snap, V_snap, W_snap = snapped
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
            
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
