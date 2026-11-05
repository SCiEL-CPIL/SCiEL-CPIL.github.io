"""Matrix multiplication 3x3 by 3x7 - rank minimization via tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact.
    Condition: for all (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _solve_for_W(n, m, p, U, V, r, tol=1e-8):
    """
    Given U and V, solve for W using least squares per (i,k) block.
    For each (i,k): W[k*n+i, :] @ (U[i*m+j, :] * V[j*p+k, :]) = 1 for all j.
    Returns W as list of lists, or None if no exact solution.
    """
    W = [[0.0] * r for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, r))
            for j in range(m):
                for c in range(r):
                    M[j, c] = U[i*m+j][c] * V[j*p+k][c]
            
            target = np.ones(m)
            try:
                w, residuals, rank_M, sv = np.linalg.lstsq(M, target, rcond=None)
            except np.linalg.LinAlgError:
                return None
            
            if residuals.size > 0:
                if residuals[0] > tol:
                    return None
            
            for j in range(m):
                s = sum(M[j, c] * w[c] for c in range(r))
                if abs(s - 1.0) > tol:
                    return None
            
            W[k*n+i] = [float(w[c]) for c in range(r)]
    
    return W

def _generate_sparse_UV(n, m, p, r, seed):
    """Generate random sparse U and V with entries in {-1, 0, 1}."""
    rng = np.random.RandomState(seed)
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    prob = 0.35
    for i in range(n * m):
        for c in range(r):
            if rng.random() < prob:
                U[i][c] = int(rng.choice([-1, 1]))
    for j in range(m * p):
        for c in range(r):
            if rng.random() < prob:
                V[j][c] = int(rng.choice([-1, 1]))
    return U, V

def _prune_columns(U, V, W, n, m, p, tol=1e-9, max_iter=50):
    """Greedy column pruning: try removing one column at a time, re-solving W."""
    r = len(U[0])
    active = list(range(r))
    for _ in range(max_iter):
        improved = False
        for idx, c in enumerate(active):
            if len(active) <= 1:
                break
            new_active = [cc for cc in active if cc != c]
            U_sub = [[U[i][cc] for cc in new_active] for i in range(len(U))]
            V_sub = [[V[i][cc] for cc in new_active] for i in range(len(V))]
            W_sub = _solve_for_W(n, m, p, U_sub, V_sub, len(new_active))
            if W_sub is not None and _verify(U_sub, V_sub, W_sub, n, m, p, tol):
                U = U_sub
                V = V_sub
                W = W_sub
                active = new_active
                improved = True
                break
        if not improved:
            break
    return U, V, W, len(active)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    school = _schoolbook(n, m, p)
    best_U = [row[:] for row in school["U"]]
    best_V = [row[:] for row in school["V"]]
    best_W = [row[:] for row in school["W"]]
    best_rank = n * m * p
    
    # Strategy: Fix U and V to random sparse patterns, solve for W via least squares,
    # then prune columns to find minimal rank.
    
    try:
        for target_rank in range(15, best_rank + 1):
            if time.time() - start_time > deadline:
                break
            
            for seed in range(30):
                if time.time() - start_time > deadline:
                    break
                
                U, V = _generate_sparse_UV(n, m, p, target_rank, seed)
                W = _solve_for_W(n, m, p, U, V, target_rank)
                if W is None:
                    continue
                
                if _verify(U, V, W, n, m, p):
                    U_pruned, V_pruned, W_pruned, pruned_rank = _prune_columns(U, V, W, n, m, p)
                    if pruned_rank < best_rank:
                        best_U = U_pruned
                        best_V = V_pruned
                        best_W = W_pruned
                        best_rank = pruned_rank
                        break
    except Exception:
        # If any error occurs during search, fall back to the best result found so far
        pass

    return {"U": best_U, "V": best_V, "W": best_W}
