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
    
    # Try to find a rank-50 decomposition using LM
    try:
        result = _try_lm_rank50(deadline)
        if result is not None:
            return result
    except Exception:
        pass
    
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
    Vectorized computation.
    """
    n, m, p = 2, 2, 15
    rank = U.shape[1]
    
    # Reshape U, V, W
    U_reshaped = U.reshape(n, m, rank)  # U[i*m+j, r]
    V_reshaped = V.reshape(m, p, rank)  # V[j*p+k, r]
    W_reshaped = W.reshape(p, n, rank)  # W[k*n+i, r]
    
    # Compute tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Vectorize over k using einsum
    # U_reshaped[i,j,:] * V_reshaped[j,k,:] * W_reshaped[k,i,:]
    # For each (i,j): sum over r of U[i,j,r] * (sum over k of V[j,k,r]*W[k,i,r])
    # Let's use einsum: 'ijr,jkr,kir->ijk'
    T_computed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    return T_computed - T

def _try_lm_rank50(deadline):
    """Try to find a rank-50 decomposition using LM optimization with multiple restarts."""
    n, m, p = 2, 2, 15
    rank = 50
    
    T = _build_tensor(n, m, p)
    
    # Initialize from rank-53, drop last 3 columns
    base = _build_rank53()
    U_base = np.array(base["U"], dtype=float)
    V_base = np.array(base["V"], dtype=float)
    W_base = np.array(base["W"], dtype=float)
    
    # Flatten to 1D vector for LM
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
    
    # Generate diverse initializations
    initializations = []
    
    # Strategy 1: Drop last 3 columns from rank-53
    U_init1 = U_base[:, :rank].copy()
    V_init1 = V_base[:, :rank].copy()
    W_init1 = W_base[:, :rank].copy()
    initializations.append(_flatten(U_init1, V_init1, W_init1))
    
    # Strategy 2: Drop first 3 columns
    U_init2 = np.delete(U_base, [0, 1, 2], axis=1)
    V_init2 = np.delete(V_base, [0, 1, 2], axis=1)
    W_init2 = np.delete(W_base, [0, 1, 2], axis=1)
    initializations.append(_flatten(U_init2, V_init2, W_init2))
    
    # Strategy 3: Drop middle columns
    U_init3 = np.delete(U_base, [25, 26, 27], axis=1)
    V_init3 = np.delete(V_base, [25, 26, 27], axis=1)
    W_init3 = np.delete(W_base, [25, 26, 27], axis=1)
    initializations.append(_flatten(U_init3, V_init3, W_init3))
    
    # Strategy 4: Random sparse initialization (mostly zeros with some structure)
    np.random.seed(123)
    for seed_val in [42, 99, 200]:
        np.random.seed(seed_val)
        U_rand = np.zeros((n*m, rank))
        V_rand = np.zeros((m*p, rank))
        W_rand = np.zeros((p*n, rank))
        # Fill with random values from {-1, 0, 1} sparsely
        for r in range(rank):
            # Each column has some non-zero entries
            n_nonzero = np.random.randint(3, 7)
            idx_u = np.random.choice(n*m, size=min(n_nonzero, n*m), replace=False)
            U_rand[idx_u, r] = np.random.choice([-1, 1], size=n_nonzero)
            
            n_nonzero_v = np.random.randint(2, 5)
            idx_v = np.random.choice(m*p, size=min(n_nonzero_v, m*p), replace=False)
            V_rand[idx_v, r] = np.random.choice([-1, 1], size=n_nonzero_v)
            
            n_nonzero_w = np.random.randint(2, 5)
            idx_w = np.random.choice(p*n, size=min(n_nonzero_w, p*n), replace=False)
            W_rand[idx_w, r] = np.random.choice([-1, 1], size=n_nonzero_w)
        
        initializations.append(_flatten(U_rand, V_rand, W_rand))
    
    best_result = None
    best_norm = np.inf
    
    # Try all initializations
    for idx, x_start in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        try:
            result = least_squares(_residual_vector, x_start, method='lm', max_nfev=3000)
            norm = np.linalg.norm(result.fun)
            if norm < best_norm:
                best_norm = norm
                best_result = result
                if norm < 1e-12:
                    break
        except Exception:
            continue
    
    if best_result is not None:
        U_opt, V_opt, W_opt = _unflatten(best_result.x)
        
        # Try multiple rounding strategies
        rounding_results = []
        
        # Strategy A: Round to half-integers
        U_r1 = np.round(U_opt * 2) / 2
        V_r1 = np.round(V_opt * 2) / 2
        W_r1 = np.round(W_opt * 2) / 2
        resid1 = _compute_residual(U_r1, V_r1, W_r1, T)
        rounding_results.append((np.max(np.abs(resid1)), U_r1, V_r1, W_r1))
        
        # Strategy B: Round to integers
        U_r2 = np.round(U_opt)
        V_r2 = np.round(V_opt)
        W_r2 = np.round(W_opt)
        resid2 = _compute_residual(U_r2, V_r2, W_r2, T)
        rounding_results.append((np.max(np.abs(resid2)), U_r2, V_r2, W_r2))
        
        # Strategy C: Round to quarter-integers then scale
        U_r3 = np.round(U_opt * 4) / 4
        V_r3 = np.round(V_opt * 4) / 4
        W_r3 = np.round(W_opt * 4) / 4
        resid3 = _compute_residual(U_r3, V_r3, W_r3, T)
        rounding_results.append((np.max(np.abs(resid3)), U_r3, V_r3, W_r3))
        
        # Pick the best rounding
        rounding_results.sort(key=lambda x: x[0])
        best_resid, U_best, V_best, W_best = rounding_results[0]
        
        if best_resid < 1e-10:
            return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
        
        # If rounding failed but continuous solution is good, do a polish pass
        if time.time() < deadline - 10 and best_norm < 1e-6:
            try:
                result2 = least_squares(_residual_vector, best_result.x, method='lm', max_nfev=10000, ftol=1e-15)
                U_opt2, V_opt2, W_opt2 = _unflatten(result2.x)
                
                # Try rounding again after polish
                for scale in [2, 4]:
                    U_r = np.round(U_opt2 * scale) / scale
                    V_r = np.round(V_opt2 * scale) / scale
                    W_r = np.round(W_opt2 * scale) / scale
                    resid_r = _compute_residual(U_r, V_r, W_r, T)
                    if np.max(np.abs(resid_r)) < 1e-10:
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            except Exception:
                pass
    
    return None
