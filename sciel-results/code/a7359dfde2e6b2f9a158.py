"""Rewritten program for 3x4x7 matrix multiplication tensor decomposition.

Strategy:
  1. Start with the schoolbook rank-84 decomposition (exact, entries in {0,1}).
  2. Attempt to reduce the rank by 1 (to 83) using a structured search:
     - Fix U and V to be the first 83 columns of the schoolbook (entries 0/1).
     - Solve for W via least squares; if the solution is exact and has
       integer/half-integer entries, accept rank 83.
  3. If that fails, try a few random rank-83 initialisations of U and V
     (entries from a small set) and solve for W.
  4. Fall back to the schoolbook rank-84 result.

The verification uses the tensor identity:
  sum_c W[k,i,c] * U[a,b,c] * V[b,d,c] == delta_{a,i} * delta_{d,k}
for all i,k,a,b,d.
"""

DIMENSIONS = [3, 4, 7]

import numpy as np
import time
import itertools

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that U, V, W form a valid decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    U3 = np.array(U).reshape(n, m, rank)
    V3 = np.array(V).reshape(m, p, rank)
    W3 = np.array(W).reshape(p, n, rank)
    # T[i,k,a,b,d] = sum_c W[k,i,c] U[a,b,c] V[b,d,c]
    T = np.einsum('kic,abc,bdc->ikabd', W3, U3, V3)
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for d in range(p):
                        expected = 1.0 if (a == i and d == k) else 0.0
                        if abs(T[i, k, a, b, d] - expected) > tol:
                            return False
    return True

def _is_nice(arr, tol=1e-9):
    """Check if all entries are close to integers or half-integers."""
    arr = np.array(arr, dtype=float)
    doubled = arr * 2.0
    rounded = np.round(doubled)
    return np.all(np.abs(doubled - rounded) < tol)

def _schoolbook(n, m, p):
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

def _solve_for_W(U, V, n, m, p, rank):
    """Given U (n*m x rank) and V (m*p x rank), solve for W (p*n x rank).
    Returns W if a valid solution exists, else None.
    """
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    
    # Number of equations: n*p*n*m*p
    n_eq = n * p * n * m * p
    n_unk = p * n * rank
    
    A = np.zeros((n_eq, n_unk))
    b = np.zeros(n_eq)
    
    eq_idx = 0
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for bidx in range(m):
                    for d in range(p):
                        if a == i and d == k:
                            b[eq_idx] = 1.0
                        base = k * n * rank + i * rank
                        # Coefficient for W[k,i,c] is U[a,bidx,c] * V[bidx,d,c]
                        for c in range(rank):
                            A[eq_idx, base + c] = U3[a, bidx, c] * V3[bidx, d, c]
                        eq_idx += 1
    
    try:
        W_flat, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    
    if len(residuals) > 0 and residuals[0] > 1e-8:
        return None
    
    W = W_flat.reshape(p, n, rank).T.reshape(p * n, rank)  # W[k*n+i, c]
    # Actually reshape: W_flat is indexed by k*n*rank + i*rank + c
    # So W_flat.reshape(p, n, rank) gives W[k, i, c], we need W[k*n+i, c]
    W = W_flat.reshape(p, n, rank).reshape(p * n, rank)
    
    return W

def _try_rank_83(n, m, p, deadline):
    """Try to find a rank-83 decomposition."""
    target_rank = n * m * p - 1  # 83
    
    # Strategy 1: Use first 83 columns of schoolbook for U and V, solve for W
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    U_try = U_sb[:, :target_rank]
    V_try = V_sb[:, :target_rank]
    W_sol = _solve_for_W(U_try, V_try, n, m, p, target_rank)
    if W_sol is not None:
        if _verify(U_try, V_try, W_sol, n, m, p, target_rank):
            if _is_nice(U_try) and _is_nice(V_try) and _is_nice(W_sol):
                return U_try, V_try, W_sol
    
    # Strategy 2: Random U and V with entries from {-1, 0, 1}, solve for W
    rng = np.random.RandomState(42)
    max_attempts = 200
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        U_rand = rng.choice([-1, 0, 1], size=(n * m, target_rank))
        V_rand = rng.choice([-1, 0, 1], size=(m * p, target_rank))
        W_sol = _solve_for_W(U_rand, V_rand, n, m, p, target_rank)
        if W_sol is not None:
            if _verify(U_rand, V_rand, W_sol, n, m, p, target_rank):
                if _is_nice(U_rand) and _is_nice(V_rand) and _is_nice(W_sol):
                    return U_rand, V_rand, W_sol
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to reduce rank
    result = _try_rank_83(n, m, p, deadline)
    if result is not None:
        U, V, W = result
        best_rank = n * m * p - 1
    
    # Convert to lists of lists as required
    return {
        "U": np.array(U).tolist(),
        "V": np.array(V).tolist(),
        "W": np.array(W).tolist()
    }

# EVOLVE-BLOCK-END
