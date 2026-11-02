"""
Matrix multiplication 2,2,15 tensor rank minimization.
Strategy: Start from rank-53 block decomposition, use LM to find dense rank-50
decomposition, then round to half-integers.
"""

import numpy as np
import time

def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # First, build the rank-53 baseline
    baseline = _build_rank53()
    
    # Try to find lower rank decompositions
    best_result = None
    best_rank = 53
    
    # Try ranks 50, 51, 52. Lower is better.
    for target_rank in [50, 51, 52]:
        if time.time() > deadline - 10: # Need at least 10s for verification and return
            break
            
        try:
            result = _try_lm_rank(target_rank, deadline)
            if result is not None:
                best_result = result
                best_rank = target_rank
                # If we found rank 50, we can stop as it's likely minimal or close to it
                # But let's see if we can find even lower? Theoretically lower bound is higher.
                # Let's break if we find 50 to save time, or continue if we want to be sure?
                # Usually finding 50 is a big jump from 53.
                if target_rank == 50:
                    break
        except Exception:
            pass
            
    if best_result is not None:
        return best_result
        
    # Fallback to baseline
    return baseline

def _build_rank53():
    n, m, p = 2, 2, 15
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    U_sc = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_sc = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_sc = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    col = 0
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        for r in range(7):
            c = col + r
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_s[i*2 + j][r]
            for j in range(2):
                V[j*p + k0][c] = V_s[j*2 + 0][r]
                V[j*p + k1][c] = V_s[j*2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk*n + i][c] = W_s[k*2 + i][r]
        col += 7
        
    k_last = 14
    for r in range(4):
        c = col + r
        for i in range(2):
            for j in range(2):
                U[i*m + j][c] = U_sc[i*2 + j][r]
        for j in range(2):
            V[j*p + k_last][c] = V_sc[j][r]
        for i in range(2):
            W[k_last*n + i][c] = W_sc[i][r]
            
    return {"U": U, "V": V, "W": W}

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if all indices valid."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _compute_residual(U, V, W, T):
    """
    Compute residual: sum_r U[:,r] outer V[:,r] outer W[:,r] vs T
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    """
    n, m, p = 2, 2, 15
    rank = U.shape[1]
    
    # Reshape U, V, W
    U_reshaped = U.reshape(n, m, rank)  # U[i*m+j, r]
    V_reshaped = V.reshape(m, p, rank)  # V[j*p+k, r]
    W_reshaped = W.reshape(p, n, rank)  # W[k*n+i, r]
    
    # Compute tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T_computed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_computed[i, j, k] = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
    
    return T_computed - T

def _try_lm_rank(rank, deadline):
    """Try to find a decomposition of the given rank using LM optimization with restarts."""
    n, m, p = 2, 2, 15
    
    T = _build_tensor(n, m, p)
    
    # Initialize from rank-53, drop last (53-rank) columns
    base = _build_rank53()
    U_init_full = np.array(base["U"], dtype=float)
    V_init_full = np.array(base["V"], dtype=float)
    W_init_full = np.array(base["W"], dtype=float)
    
    U_init = U_init_full[:, :rank]
    V_init = V_init_full[:, :rank]
    W_init = W_init_full[:, :rank]
    
    def _flatten(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def _unflatten(x):
        U_size = n * m * rank
        V_size = m * p * rank
        W_size = p * n * rank
        U = x[:U_size].reshape(n * m, rank)
        V = x[U_size:U_size+V_size].reshape(m * p, rank)
        W = x[U_size+V_size:].reshape(p * n, rank)
        return U, V, W
    
    def _residual_vector(x):
        U, V, W = _unflatten(x)
        resid = _compute_residual(U, V, W, T)
        return resid.flatten()
    
    from scipy.optimize import least_squares
    
    # Try a few restarts
    num_restarts = 3
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        if restart == 0:
            x0 = _flatten(U_init, V_init, W_init)
        else:
            # Add random noise to escape local minima
            noise_scale = 0.1
            U_noisy = U_init + rng.normal(0, noise_scale, U_init.shape)
            V_noisy = V_init + rng.normal(0, noise_scale, V_init.shape)
            W_noisy = W_init + rng.normal(0, noise_scale, W_init.shape)
            x0 = _flatten(U_noisy, V_noisy, W_noisy)
            
        try:
            result = least_squares(_residual_vector, x0, method='lm', max_nfev=5000)
            
            if result.success or np.linalg.norm(result.fun) < 1e-8:
                U_opt, V_opt, W_opt = _unflatten(result.x)
                
                # Try to round to half-integers
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                # Verify
                resid = _compute_residual(U_rounded, V_rounded, W_rounded, T)
                if np.max(np.abs(resid)) < 1e-10:
                    # Convert to lists
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
        except Exception:
            continue
            
    return None
