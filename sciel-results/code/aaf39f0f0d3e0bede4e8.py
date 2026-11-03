"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify exact tensor identity: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta(i,j,k)."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (i, j, k) == (0, 0, 0) else 0.0
                u_vals = U_arr[i * m + j, :]
                v_vals = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                total = float(np.dot(np.multiply(u_vals, v_vals), w_val))
                if abs(total - expected) > 1e-5:
                    return False
    return True

def _tensor_residual(U_arr, V_arr, W_arr, n, m, p):
    """Return flattened residual vector: for each (i,j,k), sum_c U*V*W - delta(i,j,k)."""
    nm = n * m
    mp = m * p
    pn = p * n
    r = U_arr.shape[1]
    residual = np.zeros(n * m * p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                u_vals = U_arr[i * m + j, :]
                v_vals = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                residual[idx] = float(np.dot(np.multiply(u_vals, v_vals), w_val))
                if (i, j, k) == (0, 0, 0):
                    residual[idx] -= 1.0
    return residual

def _random_init(n, m, p, rank, rng):
    """Initialize random factor matrices with small random values."""
    U = rng.normal(0, 0.5, size=(n * m, rank))
    V = rng.normal(0, 0.5, size=(m * p, rank))
    W = rng.normal(0, 0.5, size=(p * n, rank))
    return U, V, W

def _levmarquardt(U, V, W, n, m, p, max_iter=2000, tol=1e-9, lam_start=1e-3, deadline=None):
    """
    Levenberg-Marquardt optimization of factor matrices to minimize tensor residual.
    Returns (U, V, W, converged).
    """
    nm = n * m
    mp = m * p
    pn = p * n
    r = U.shape[1]
    
    # Flatten all parameters
    U_flat = U.flatten().copy()
    V_flat = V.flatten().copy()
    W_flat = W.flatten().copy()
    x = np.concatenate([U_flat, V_flat, W_flat])
    x_len = len(x)
    
    U_size = nm * r
    V_size = mp * r
    W_size = pn * r
    
    def _unpack(x):
        U_f = x[:U_size].reshape(nm, r)
        V_f = x[U_size:U_size+V_size].reshape(mp, r)
        W_f = x[U_size+V_size:].reshape(pn, r)
        return U_f, V_f, W_f
    
    def _residual(x):
        U_f, V_f, W_f = _unpack(x)
        return _tensor_residual(U_f, V_f, W_f, n, m, p)
    
    lam = lam_start
    current_res = _residual(x)
    current_ssr = float(np.dot(current_res, current_res))
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute Jacobian numerically (sparse-ish: each parameter affects few residual entries)
        # For efficiency, use finite differences with central difference
        J = np.zeros((n * m * p, x_len))
        step = 1e-7
        for col in range(x_len):
            x_pert = x.copy()
            x_pert[col] += step
            res_pert = _residual(x_pert)
            J[:, col] = (res_pert - current_res) / step
        
        # Normal equations: (J^T J + lam * diag(J^T J)) dx = -J^T r
        JtJ = J.T @ J
        JtR = J.T @ current_res
        diag = np.diag(JtJ).copy()
        diag[diag < 1e-12] = 1e-12
        
        for trial in range(10):
            A = JtJ.copy()
            np.fill_diagonal(A, diag * (1 + lam))
            try:
                dx = np.linalg.solve(A, -JtR)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            
            x_new = x + dx
            res_new = _residual(x_new)
            ssr_new = float(np.dot(res_new, res_new))
            
            if ssr_new < current_ssr:
                x = x_new
                current_res = res_new
                current_ssr = ssr_new
                lam *= 0.3
                if lam < 1e-12:
                    lam = 1e-12
                if current_ssr < tol:
                    break
            else:
                lam *= 10
                if lam > 1e10:
                    break
        
        if current_ssr < tol:
            break
    
    U_final, V_final, W_final = _unpack(x)
    converged = current_ssr < 1e-6
    return U_final, V_final, W_final, converged

def _schoolbook_init(n, m, p):
    """Return schoolbook rank n*m*p factor matrices."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_random_low_rank(n, m, p, target_rank, rng, deadline):
    """
    Try to find a rank-target_rank decomposition via random init + LM.
    Returns (U, V, W, success).
    """
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize: start from schoolbook, merge random pairs of columns
        # to reduce rank, then add small noise to escape bad local minima
        U0, V0, W0 = _schoolbook_init(n, m, p)
        full_rank = n * m * p
        
        if target_rank >= full_rank:
            return U0, V0, W0, True
        
        # Randomly select which columns to keep/merge
        # Strategy: randomly assign each of the full_rank schoolbook terms
        # to one of target_rank "slots", summing contributions
        slot_assign = rng.randint(0, target_rank - 1, size=full_rank)
        
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    slot = slot_assign[col]
                    U[i * m + j, slot] += 1.0
                    V[j * p + k, slot] += 1.0
                    W[k * n + i, slot] += 1.0
                    col += 1
        
        # Add small random perturbation to break symmetries
        noise_scale = 0.1
        U += rng.normal(0, noise_scale, size=U.shape)
        V += rng.normal(0, noise_scale, size=V.shape)
        W += rng.normal(0, noise_scale, size=W.shape)
        
        # Run Levenberg-Marquardt
        U_opt, V_opt, W_opt, converged = _levmarquardt(
            U, V, W, n, m, p,
            max_iter=500, tol=1e-10, lam_start=1e-3,
            deadline=deadline
        )
        
        if converged:
            # Verify with tighter tolerance
            if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p):
                return U_opt, V_opt, W_opt, True
    
    return None, None, None, False

def construct():
    t0 = time.time()
    DEADLINE = 120.0
    MARGIN = 15.0
    effective_deadline = t0 + DEADLINE - MARGIN
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    full_rank = n * m * p  # 75
    
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    # Start with schoolbook (guaranteed valid)
    U_best, V_best, W_best = _schoolbook_init(n, m, p)
    best_rank = full_rank
    
    # Try to find lower-rank decompositions
    # Search from lower ranks upward, but prioritize finding ANY valid low rank
    # Known lower bound: max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # Also, for matrix multiplication, rank >= 2*max(n,p)*min(n,p) - something...
    # For 3x5x5, let's try ranks from 30 up to 74
    
    # First, try a few random initializations at various ranks
    ranks_to_try = list(range(30, 75))
    # Shuffle to try diverse ranks
    random.shuffle(ranks_to_try)
    
    # But prioritize lower ranks (better score)
    # Sort by rank ascending but with some randomness
    ranks_to_try = sorted(ranks_to_try, key=lambda r: (r, rng.random()))
    
    for target_rank in ranks_to_try:
        if time.time() > effective_deadline:
            break
        
        if target_rank >= best_rank:
            continue
        
        U_new, V_new, W_new, success = _try_random_low_rank(
            n, m, p, target_rank, rng, effective_deadline
        )
        
        if success and U_new is not None:
            best_rank = target_rank
            U_best = U_new
            V_best = V_new
            W_best = W_new
            
            # If we found a low rank, try to go even lower
            # But don't waste too much time
            if target_rank <= 40:
                # Try a few more restarts at this rank
                for extra in range(3):
                    if time.time() > effective_deadline:
                        break
                    U_extra, V_extra, W_extra, extra_success = _try_random_low_rank(
                        n, m, p, target_rank, rng, effective_deadline
                    )
                    if extra_success and U_extra is not None:
                        U_best = U_extra
                        V_best = V_extra
                        W_best = W_extra
    
    # Final verification
    if not _verify(U_best.tolist(), V_best.tolist(), W_best.tolist(), n, m, p):
        # Fallback to schoolbook
        U_best, V_best, W_best = _schoolbook_init(n, m, p)
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}

# EVOLVE-BLOCK-END
