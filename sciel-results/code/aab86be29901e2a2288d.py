"""Matrix multiplication 3,5,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using rational arithmetic."""
    r = len(U[0])
    Uf = [[Fraction(U[i][c]) for c in range(r)] for i in range(n*m)]
    Vf = [[Fraction(V[j][c]) for c in range(r)] for j in range(m*p)]
    Wf = [[Fraction(W[k][c]) for c in range(r)] for k in range(p*n)]
    
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = Fraction(0)
                    for c in range(r):
                        s += Wf[k*n+i][c] * Uf[i*m+j1][c] * Vf[j2*p+k][c]
                    target = 1 if j1 == j2 else 0
                    if s != target:
                        return False
    return True

def _verify_float(U, V, W, n, m, p, tol=1e-8):
    """Quick float verification."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0.0
                    for c in range(r):
                        s += W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c]
                    target = 1.0 if j1 == j2 else 0.0
                    if abs(s - target) > tol:
                        return False
    return True

def _als_optimize(U, V, W, n, m, p, r, max_iters=200, deadline=None, tol=1e-10):
    """ALS optimization with convergence check, vectorized."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    prev_residual = np.inf
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update W: for each (k,i), solve A_W @ w = b where A_W rows are U[i*m+j1]*V[j2*p+k]
        for k in range(p):
            for i in range(n):
                # Build A_W as (m*m, r): row (j1,j2) = U[i*m+j1] * V[j2*p+k]
                U_i = U[i*m:(i+1)*m, :]  # (m, r)
                V_k = V[k::p, :]  # (m, r) -- rows j2*p+k for j2 in 0..m-1
                # A_W[j1*m+j2] = U_i[j1] * V_k[j2] (elementwise)
                # Use outer product structure: A_W = (U_i[:, None, :] * V_k[None, :, :]).reshape(m*m, r)
                A_W = (U_i[:, None, :] * V_k[None, :, :]).reshape(m*m, r)
                b = np.zeros(m*m)
                b[::m+1] = 1.0  # diagonal entries
                w, *_ = np.linalg.lstsq(A_W, b, rcond=None)
                W[k*n+i, :] = w
        
        # Update U: for each (i,j1), solve A_U @ u = b
        for i in range(n):
            for j1 in range(m):
                # A_U rows indexed by (k,j2): W[k*n+i] * V[j2*p+k]
                # b[k*m+j2] = 1 if j1==j2 else 0
                A_U = np.zeros((p*m, r))
                b = np.zeros(p*m)
                for k in range(p):
                    W_ik = W[k*n+i, :]
                    V_k = V[k::p, :]  # (m, r)
                    A_U[k*m:(k+1)*m, :] = W_ik[None, :] * V_k
                    b[k*m+j1] = 1.0
                u, *_ = np.linalg.lstsq(A_U, b, rcond=None)
                U[i*m+j1, :] = u
        
        # Update V: for each (j2,k), solve A_V @ v = b
        for j2 in range(m):
            for k in range(p):
                # A_V rows indexed by (i,j1): W[k*n+i] * U[i*m+j1]
                # b[i*m+j1] = 1 if j1==j2 else 0
                A_V = np.zeros((n*m, r))
                b = np.zeros(n*m)
                for i in range(n):
                    W_ik = W[k*n+i, :]
                    U_i = U[i*m:(i+1)*m, :]  # (m, r)
                    A_V[i*m:(i+1)*m, :] = W_ik[None, :] * U_i
                    b[i*m+j2] = 1.0
                v, *_ = np.linalg.lstsq(A_V, b, rcond=None)
                V[j2*p+k, :] = v
        
        # Check convergence
        residual = _compute_residual(U, V, W, n, m, p)
        if abs(residual - prev_residual) < tol:
            break
        prev_residual = residual
    
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """Compute the residual of the bilinear identity."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0.0
                    for c in range(r):
                        s += W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c]
                    target = 1.0 if j1 == j2 else 0.0
                    max_err = max(max_err, abs(s - target))
    return max_err

def _round_to_rational(arr, max_denom=32):
    """Round array entries to rationals with small denominators."""
    result = []
    for i in range(len(arr)):
        row = []
        for c in range(len(arr[0])):
            val = arr[i][c]
            best = round(val)
            best_err = abs(val - best)
            # If max_denom is 1, we only try integers
            if max_denom > 1:
                for d in range(2, max_denom+1):
                    num = round(val * d)
                    cand = num / d
                    err = abs(val - cand)
                    if err < best_err:
                        best = cand
                        best_err = err
            row.append(best)
        result.append(row)
    return result

def _polish(U, V, W, n, m, p, r, deadline, max_iters=50, tol=1e-12):
    """High-precision polish: run ALS with tighter tolerance."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update W
        for k in range(p):
            for i in range(n):
                U_i = U[i*m:(i+1)*m, :]
                V_k = V[k::p, :]
                A_W = (U_i[:, None, :] * V_k[None, :, :]).reshape(m*m, r)
                b = np.zeros(m*m)
                b[::m+1] = 1.0
                w, *_ = np.linalg.lstsq(A_W, b, rcond=None)
                W[k*n+i, :] = w
        
        # Update U
        for i in range(n):
            for j1 in range(m):
                A_U = np.zeros((p*m, r))
                b = np.zeros(p*m)
                for k in range(p):
                    W_ik = W[k*n+i, :]
                    V_k = V[k::p, :]
                    A_U[k*m:(k+1)*m, :] = W_ik[None, :] * V_k
                    b[k*m+j1] = 1.0
                u, *_ = np.linalg.lstsq(A_U, b, rcond=None)
                U[i*m+j1, :] = u
        
        # Update V
        for j2 in range(m):
            for k in range(p):
                A_V = np.zeros((n*m, r))
                b = np.zeros(n*m)
                for i in range(n):
                    W_ik = W[k*n+i, :]
                    U_i = U[i*m:(i+1)*m, :]
                    A_V[i*m:(i+1)*m, :] = W_ik[None, :] * U_i
                    b[i*m+j2] = 1.0
                v, *_ = np.linalg.lstsq(A_V, b, rcond=None)
                V[j2*p+k, :] = v
        
        residual = _compute_residual(U, V, W, n, m, p)
        if residual < tol:
            break
    
    return U, V, W

def _try_rank_search(n, m, p, r, deadline, n_restarts=10):
    """Try to find a valid decomposition at rank r."""
    # Use different seeds for diversity
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        seed = 42 + r * 1000 + restart
        np.random.seed(seed)
        
        # Random initialization with larger scale to escape local minima
        scale = 1.0
        U = np.random.randn(n*m, r) * scale
        V = np.random.randn(m*p, r) * scale
        W = np.random.randn(p*n, r) * scale
        
        # ALS optimization
        U, V, W = _als_optimize(U, V, W, n, m, p, r, max_iters=150, deadline=deadline)
        
        # Check if close to exact
        if _verify_float(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-4):
            # Polish with tighter tolerance
            U, V, W = _polish(U, V, W, n, m, p, r, deadline, max_iters=50, tol=1e-12)
            
            # Check again
            if _verify_float(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-5):
                # Try rounding to rationals with increasing denominators
                # Start with integers (denom 1) as they are most likely for small ranks
                for max_denom in [1, 2, 4, 8, 16, 32]:
                    U_r = _round_to_rational(U.tolist(), max_denom=max_denom)
                    V_r = _round_to_rational(V.tolist(), max_denom=max_denom)
                    W_r = _round_to_rational(W.tolist(), max_denom=max_denom)
                    
                    if _verify_exact(U_r, V_r, W_r, n, m, p):
                        return U_r, V_r, W_r
    
    return None

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 135.0  # safety margin before 150s kill
    
    # Schoolbook baseline
    rank0 = n * m * p
    U0 = [[0]*rank0 for _ in range(n*m)]
    V0 = [[0]*rank0 for _ in range(m*p)]
    W0 = [[0]*rank0 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j][col] = 1
                V0[j*p+k][col] = 1
                W0[k*n+i][col] = 1
                col += 1
    
    best_rank = rank0
    best_U, best_V, best_W = U0, V0, W0
    
    # Lower bound: max(n*m, m*p, n*p) = max(15, 25, 15) = 25
    lower_bound = max(n*m, m*p, n*p)
    
    # Try ranks from lower_bound up to rank0-1
    # Focus on a smaller range first to increase depth of search
    # Lower bound is 25. Schoolbook is 75.
    # Try 25 to 40 first with many restarts.
    
    ranks_to_try = list(range(lower_bound, min(rank0, lower_bound + 20)))
    
    # Prioritize ranks near the lower bound
    ranks_to_try.sort(key=lambda r: abs(r - lower_bound))
    
    for r in ranks_to_try:
        if time.time() > deadline - 3:
            break
        
        # Adaptive number of restarts based on rank
        if r <= lower_bound + 5:
            n_restarts = 30
        elif r <= lower_bound + 15:
            n_restarts = 15
        else:
            n_restarts = 10
        
        result = _try_rank_search(n, m, p, r, deadline, n_restarts=n_restarts)
        
        if result is not None:
            U_r, V_r, W_r = result
            if r < best_rank:
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                # If we found a good one, we can break or continue to try lower
                # But since we iterate from low to high, we found the lowest so far.
                # We can break if we hit the lower bound.
                if r == lower_bound:
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
