"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

import time

# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize

def _fast_verify(U, V, W, n=2, m=2, p=6, tol=1e-9):
    """Verify tensor decomposition using vectorized NumPy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    a_idx = np.arange(n * m)
    b_idx = np.arange(m * p)
    c_idx = np.arange(p * n)
    
    j_a = a_idx % m
    jp_b = b_idx // p
    k_b = b_idx % p
    kp_c = c_idx // n
    
    T_expected = ((j_a[:, None, None] == jp_b[None, :, None]) & 
                  (k_b[None, :, None] == kp_c[None, None, :])).astype(float)
    
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    
    max_err = float(np.max(np.abs(T_computed - T_expected)))
    return max_err < tol, max_err

def _verify_decomposition(U, V, W, n=2, m=2, p=6, tol=1e-9):
    """Verify that the tensor decomposition is exact (compatibility wrapper)."""
    return _fast_verify(U, V, W, n, m, p, tol)

def _strassen_arrays(n=2, m=2, p=6):
    """Return Strassen rank-21 solution as numpy arrays."""
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        w_c00 = k0 * n + 0
        w_c10 = k0 * n + 1
        w_c01 = k1 * n + 0
        w_c11 = k1 * n + 1
        
        r = b * 7 + 0
        U[0, r] = 1; U[3, r] = 1
        V[v_e, r] = 1; V[v_h, r] = 1
        W[w_c00, r] = 1; W[w_c11, r] = 1
        
        r = b * 7 + 1
        U[2, r] = 1; U[3, r] = 1
        V[v_e, r] = 1
        W[w_c10, r] = 1; W[w_c11, r] = -1
        
        r = b * 7 + 2
        U[0, r] = 1
        V[v_f, r] = 1; V[v_h, r] = -1
        W[w_c01, r] = 1; W[w_c11, r] = 1
        
        r = b * 7 + 3
        U[3, r] = 1
        V[v_g, r] = 1; V[v_e, r] = -1
        W[w_c00, r] = 1; W[w_c10, r] = 1
        
        r = b * 7 + 4
        U[0, r] = 1; U[1, r] = 1
        V[v_h, r] = 1
        W[w_c00, r] = -1; W[w_c01, r] = 1
        
        r = b * 7 + 5
        U[2, r] = 1; U[0, r] = -1
        V[v_e, r] = 1; V[v_f, r] = 1
        W[w_c11, r] = 1
        
        r = b * 7 + 6
        U[1, r] = 1; U[3, r] = -1
        V[v_g, r] = 1; V[v_h, r] = 1
        W[w_c00, r] = 1
    
    return U, V, W

def _try_rank(rank, n=2, m=2, p=6, max_iters=50000, tol=1e-9, time_budget=50.0):
    """Try to find a rank-`rank` decomposition using vectorized optimization."""
    start_time = time.time()
    
    num_vars = rank * (n * m + m * p + p * n)
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute expected tensor
    a_idx = np.arange(nm)
    b_idx = np.arange(mp)
    c_idx = np.arange(pn)
    j_a = a_idx % m
    jp_b = b_idx // p
    k_b = b_idx % p
    kp_c = c_idx // n
    T_expected = ((j_a[:, None, None] == jp_b[None, :, None]) & 
                  (k_b[None, :, None] == kp_c[None, None, :])).astype(float)
    
    def objective(x):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            return 1e10
        
        U = x[:rank * nm].reshape(nm, rank)
        V = x[rank * nm : rank * nm + rank * mp].reshape(mp, rank)
        W = x[rank * nm + rank * mp :].reshape(pn, rank)
        
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((T_computed - T_expected) ** 2)
        return float(err)
    
    # Build initial guess
    if rank >= 21:
        U21, V21, W21 = _strassen_arrays(n, m, p)
        U_init = U21[:, :rank]
        V_init = V21[:, :rank]
        W_init = W21[:, :rank]
    else:
        # Start from Strassen with one term removed, then optimize
        U21, V21, W21 = _strassen_arrays(n, m, p)
        # Try removing each term and see which gives best initial residual
        best_init = None
        best_err = float('inf')
        for drop_idx in range(21):
            U_t = np.delete(U21, drop_idx, axis=1)
            V_t = np.delete(V21, drop_idx, axis=1)
            W_t = np.delete(W21, drop_idx, axis=1)
            # Pad to desired rank with small random values
            pad_size = rank - 20
            if pad_size > 0:
                np.random.seed(drop_idx)
                U_pad = np.random.randn(nm, pad_size) * 0.1
                V_pad = np.random.randn(mp, pad_size) * 0.1
                W_pad = np.random.randn(pn, pad_size) * 0.1
                U_t = np.hstack([U_t, U_pad])
                V_t = np.hstack([V_t, V_pad])
                W_t = np.hstack([W_t, W_pad])
            else:
                # Need to drop more terms
                # Just take first 'rank' columns
                U_t = U21[:, :rank]
                V_t = V21[:, :rank]
                W_t = W21[:, :rank]
            
            x_t = np.concatenate([U_t.flatten(), V_t.flatten(), W_t.flatten()])
            err_t = objective(x_t)
            if err_t < best_err:
                best_err = err_t
                best_init = x_t
        
        x0 = best_init
    if rank >= 21:
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    # Try L-BFGS-B
    try:
        result = minimize(objective, x0, method='L-BFGS-B',
                         options={'maxiter': max_iters, 'maxfun': max_iters, 'ftol': 1e-16, 'gtol': 1e-14})
        
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            return None
        
        final_err = result.fun
        if final_err < 1e-12:
            U_sol = result.x[:rank * nm].reshape(nm, rank)
            V_sol = result.x[rank * nm : rank * nm + rank * mp].reshape(mp, rank)
            W_sol = result.x[rank * nm + rank * mp :].reshape(pn, rank)
            
            # Try rounding to half-integers
            for scale in [2, 4, 8]:
                U_round = np.round(U_sol * scale) / scale
                V_round = np.round(V_sol * scale) / scale
                W_round = np.round(W_sol * scale) / scale
                valid, max_err = _fast_verify(U_round, V_round, W_round, n, m, p, tol=1e-9)
                if valid:
                    return U_round.tolist(), V_round.tolist(), W_round.tolist()
            
            # Integer rounding
            U_round2 = np.round(U_sol)
            V_round2 = np.round(V_sol)
            W_round2 = np.round(W_sol)
            valid2, _ = _fast_verify(U_round2, V_round2, W_round2, n, m, p, tol=1e-9)
            if valid2:
                return U_round2.tolist(), V_round2.tolist(), W_round2.tolist()
            
            if final_err < 1e-8:
                return U_sol.tolist(), V_sol.tolist(), W_sol.tolist()
        
        return None
    except Exception:
        return None

def _strassen_solution(n=2, m=2, p=6):
    """Return the known rank-21 Strassen-based solution."""
    U, V, W = _strassen_arrays(n, m, p)
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Get the known rank-21 solution as fallback
    U21, V21, W21 = _strassen_solution(n, m, p)
    best_solution = (U21, V21, W21)
    best_rank = 21
    
    # Try rank 20 with vectorized optimization
    if time.time() < deadline - 10:
        result = _try_rank(20, n, m, p, max_iters=30000, time_budget=60.0)
        if result is not None:
            best_solution = result
            best_rank = 20
    
    # Try rank 19 if time permits
    if time.time() < deadline - 10 and best_rank == 21:
        result = _try_rank(19, n, m, p, max_iters=20000, time_budget=40.0)
        if result is not None:
            best_solution = result
            best_rank = 19
    
    U, V, W = best_solution
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
