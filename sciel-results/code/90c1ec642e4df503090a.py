"""Matrix multiplication 3x3x5: rank-minimal exact tensor decomposition.

We search for an exact decomposition of the tensor
    T[i,j,k] = 1  iff  (i,j,k) valid, with the output index mapped as W[k*n+i, r]
using integer coefficients, trying ranks 41-44.  If none is found
within a strict wall-clock budget, we fall back to the provably correct schoolbook
rank 45.

Method:
  1. Use a structured local search with better initialization and more aggressive
     multi-variable updates.
  2. Use local search with random restarts.
  3. Time guard: return the best valid exact solution found, else the schoolbook rank 45.
"""

import time
import sys
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _compute_residual(U, V, W, n, m, p, rank):
    """Compute the residual tensor R where R[i,j,k] = 1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    R = np.ones((n, m, p), dtype=np.float64)
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val != 0:
                    v_slice = v_col[j * p:(j + 1) * p]
                    w_slice = w_col[i::n]
                    R[i, j, :] -= u_val * v_slice * w_slice
    return R

def _verify_solution(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact."""
    R = _compute_residual(U, V, W, n, m, p, rank)
    return np.max(np.abs(R)) < 1e-9

def _local_search_rank_r(n, m, p, rank, rng, deadline, initial=None, max_iters=2000, step_size=3):
    """
    Local search for a rank-r decomposition.
    """
    N = n * m
    P = m * p
    Q = p * n
    
    if initial is not None:
        U = initial[0].copy()
        V = initial[1].copy()
        W = initial[2].copy()
    else:
        U = rng.integers(-2, 3, size=(N, rank)).astype(np.float64)
        V = rng.integers(-2, 3, size=(P, rank)).astype(np.float64)
        W = rng.integers(-2, 3, size=(Q, rank)).astype(np.float64)

    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for _iter in range(max_iters):
        if time.time() >= deadline:
            break
            
        R = _compute_residual(U, V, W, n, m, p, rank)
        err = np.abs(R)
        max_err = np.max(err)
        
        if max_err < 1e-9:
            return (U.astype(int), V.astype(int), W.astype(int))
        
        if max_err < best_err:
            best_err = max_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Pick a random non-zero residual element
        nonzero_idx = np.argwhere(err > 1e-9)
        if len(nonzero_idx) == 0:
             return (U.astype(int), V.astype(int), W.astype(int))
        
        ri, rj, rk = nonzero_idx[rng.integers(0, len(nonzero_idx))]
        target = 1.0
        current = 0.0
        for r in range(rank):
            current += U[ri * m + rj, r] * V[rj * p + rk, r] * W[rk * n + ri, r]
            
        deficit = target - current
        
        # Try to adjust one or two entries
        # Choose a random rank r
        r = rng.integers(0, rank)
        
        uv = U[ri * m + rj, r]
        vv = V[rj * p + rk, r]
        ww = W[rk * n + ri, r]
        
        # We can adjust U, V, or W
        choice = rng.integers(0, 3)
        
        improved = False
        
        if choice == 0:
            if vv * ww != 0:
                delta = deficit / (vv * ww)
                new_val = uv + delta
                rounded = int(round(new_val))
                if abs(rounded - uv) <= step_size:
                    U[ri * m + rj, r] = rounded
                    # Check if improved
                    new_current = 0.0
                    for r2 in range(rank):
                        new_current += U[ri * m + rj, r2] * V[rj * p + rk, r2] * W[rk * n + ri, r2]
                    if abs(new_current - target) < abs(current - target) + 1e-12:
                        improved = True
                    else:
                        U[ri * m + rj, r] = uv
        elif choice == 1:
            if uv * ww != 0:
                delta = deficit / (uv * ww)
                new_val = vv + delta
                rounded = int(round(new_val))
                if abs(rounded - vv) <= step_size:
                    V[rj * p + rk, r] = rounded
                    new_current = 0.0
                    for r2 in range(rank):
                        new_current += U[ri * m + rj, r2] * V[rj * p + rk, r2] * W[rk * n + ri, r2]
                    if abs(new_current - target) < abs(current - target) + 1e-12:
                        improved = True
                    else:
                        V[rj * p + rk, r] = vv
        else:
            if uv * vv != 0:
                delta = deficit / (uv * vv)
                new_val = ww + delta
                rounded = int(round(new_val))
                if abs(rounded - ww) <= step_size:
                    W[rk * n + ri, r] = rounded
                    new_current = 0.0
                    for r2 in range(rank):
                        new_current += U[ri * m + rj, r2] * V[rj * p + rk, r2] * W[rk * n + ri, r2]
                    if abs(new_current - target) < abs(current - target) + 1e-12:
                        improved = True
                    else:
                        W[rk * n + ri, r] = ww
        
        # Occasionally do a larger random perturbation to escape local minima
        if not improved and rng.random() < 0.1:
            r_pert = rng.integers(0, rank)
            choice_pert = rng.integers(0, 3)
            if choice_pert == 0:
                U[rng.integers(0, N), r_pert] += rng.integers(-1, 2)
            elif choice_pert == 1:
                V[rng.integers(0, P), r_pert] += rng.integers(-1, 2)
            else:
                W[rng.integers(0, Q), r_pert] += rng.integers(-1, 2)

    R = _compute_residual(U, V, W, n, m, p, rank)
    if np.max(np.abs(R)) < 1e-9:
        return (U.astype(int), V.astype(int), W.astype(int))
    
    # Check best found
    R_best = _compute_residual(best_U, best_V, best_W, n, m, p, rank)
    if np.max(np.abs(R_best)) < 1e-9:
        return (best_U.astype(int), best_V.astype(int), best_W.astype(int))
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """Return a rank-minimal exact tensor decomposition."""
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s limit

    rng = np.random.default_rng(42)

    # Try rank 41 first, as it is the known minimum
    ranks_to_try = [41, 42, 43, 44]
    
    for rank in ranks_to_try:
        if time.time() >= deadline:
            break
        
        # Allocate time based on rank
        time_per_rank = 35.0 if rank == 41 else 20.0
        rank_deadline = min(deadline, time.time() + time_per_rank)
        
        # Multiple restarts
        max_restarts = 8
        for restart in range(max_restarts):
            if time.time() >= rank_deadline:
                break
                
            sol = _local_search_rank_r(n, m, p, rank, rng, rank_deadline, max_iters=1000, step_size=3)
            if sol is not None:
                U, V, W = sol
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist(),
                }

    # Fall back to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
