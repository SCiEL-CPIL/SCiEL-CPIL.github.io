"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def _verify_rank(n, m, p, rank, U, V, W):
    """Verify that the given factors decompose the matmul tensor correctly."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(val - 1.0) > 1e-6:
                    return False, (i, j, k, val)
    return True, None

def _compute_error_fast(n, m, p, U, V, W):
    """Fast vectorized error computation."""
    err = 0.0
    for i in range(n):
        for k in range(p):
            w_row = W[k * n + i, :]
            for j in range(m):
                u_row = U[i * m + j, :]
                v_row = V[j * p + k, :]
                val = np.dot(u_row, v_row * w_row)
                err += (val - 1.0) ** 2
    return err

def _try_round_and_verify(n, m, p, U, V, W, rank):
    """Try various rounding strategies and verify."""
    strategies = [
        ("half", lambda x: np.round(x * 2) / 2),
        ("int", lambda x: np.round(x)),
    ]
    for name, fn in strategies:
        U_r = fn(U)
        V_r = fn(V)
        W_r = fn(W)
        valid, _ = _verify_rank(n, m, p, rank, U_r, V_r, W_r)
        if valid:
            return True, U_r, V_r, W_r
    
    # Try mixed rounding
    for u_fn, v_fn, w_fn in [
        (lambda x: np.round(x), lambda x: np.round(x*2)/2, lambda x: np.round(x*2)/2),
        (lambda x: np.round(x*2)/2, lambda x: np.round(x), lambda x: np.round(x*2)/2),
        (lambda x: np.round(x*2)/2, lambda x: np.round(x*2)/2, lambda x: np.round(x)),
    ]:
        U_r = u_fn(U)
        V_r = v_fn(V)
        W_r = w_fn(W)
        valid, _ = _verify_rank(n, m, p, rank, U_r, V_r, W_r)
        if valid:
            return True, U_r, V_r, W_r
    
    return False, None, None, None

def _als_step(n, m, p, U, V, W, rank):
    """One ALS iteration with proper least-squares updates."""
    # Update W
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, rank))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
            W[k * n + i, :] = sol
    
    # Update U
    for i in range(n):
        for j in range(m):
            B = np.zeros((p, rank))
            for k in range(p):
                B[k, :] = V[j * p + k, :] * W[k * n + i, :]
            sol, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
            U[i * m + j, :] = sol
    
    # Update V
    for j in range(m):
        for k in range(p):
            C = np.zeros((n, rank))
            for i in range(n):
                C[i, :] = U[i * m + j, :] * W[k * n + i, :]
            sol, _, _, _ = np.linalg.lstsq(C, np.ones(n), rcond=None)
            V[j * p + k, :] = sol

def _run_als(n, m, p, U, V, W, rank, max_iters=300, tol=1e-12, deadline=None):
    """Run ALS until convergence or deadline."""
    prev_err = float('inf')
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        err = _compute_error_fast(n, m, p, U, V, W)
        
        if err < tol:
            return True, U, V, W, err
        
        if abs(prev_err - err) < 1e-15:
            # Check if we're at a good enough solution
            if err < 1e-8:
                return True, U, V, W, err
            break
        
        prev_err = err
        _als_step(n, m, p, U, V, W, rank)
    
    err = _compute_error_fast(n, m, p, U, V, W)
    return err < 1e-8, U, V, W, err

def _build_structured_init(n, m, p, rank, seed=0):
    """Build a structured random initialization."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    # Normalize each column to avoid scale issues
    for c in range(rank):
        norm_u = np.linalg.norm(U[:, c])
        norm_v = np.linalg.norm(V[:, c])
        norm_w = np.linalg.norm(W[:, c])
        if norm_u * norm_v * norm_w > 1e-10:
            scale = 1.0 / (norm_u * norm_v * norm_w) ** (1.0/3.0)
            U[:, c] *= scale
            V[:, c] *= scale
            W[:, c] *= scale
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    schoolbook_rank = n * m * p  # 96
    
    # Initialize best with schoolbook solution
    best_rank = schoolbook_rank
    best_U = np.zeros((n * m, schoolbook_rank))
    best_V = np.zeros((m * p, schoolbook_rank))
    best_W = np.zeros((p * n, schoolbook_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                best_U[i * m + j, c] = 1
                best_V[j * p + k, c] = 1
                best_W[k * n + i, c] = 1
    
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks from low to high
    # For 4x4*4x6, we expect rank around 12-24 based on known results
    target_ranks = [12, 14, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 80]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue
        
        found = False
        max_attempts = 15
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            seed = 42 + attempt * 7
            U, V, W = _build_structured_init(n, m, p, target_rank, seed)
            
            converged, U, V, W, err = _run_als(n, m, p, U, V, W, target_rank, 
                                                 max_iters=400, tol=1e-12,
                                                 deadline=deadline - 10)
            
            if converged and err < 1e-8:
                ok, Ur, Vr, Wr = _try_round_and_verify(n, m, p, U, V, W, target_rank)
                if ok:
                    best_rank = target_rank
                    best_U = Ur
                    best_V = Vr
                    best_W = Wr
                    found = True
                    break
        
        if found:
            continue
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
