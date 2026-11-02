"""
Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition.

We use the schoolbook decomposition of rank 60 as a safe fallback,
and attempt to find a lower-rank decomposition via randomized rank-revealing
search and structured ansatz.

The matrix multiplication tensor T has entries T[i,j,k] = 1 if we multiply A[i,j]*B[j,k].
A decomposition of rank r gives U (12 x r), V (20 x r), W (15 x r) such that
T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  (with W indexed as W[k*n+i, c]).

We try to find r < 60. Known lower bounds: r >= max(n*m, m*p, p*n) = max(12,20,15)=20.
Upper bound: 60 (schoolbook).

For 3x4x5, we attempt a structured search:
1. Start from schoolbook and try to merge columns.
2. Try randomized initialization with gradient-free refinement.
3. Use the fact that the tensor is sparse to do rank-revealing via flattening.

Given the 150s deadline, we do:
- Phase 1: Try to find a decomposition of rank r for r from 20 to 59 by
  setting up the bilinear system and solving via SVD on the unfolding.
- Phase 2: If a candidate is found, verify exactly.
- Phase 3: Return best found (schoolbook if nothing better).
"""

import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 5]

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U,V,W exactly decompose the matmul tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    r = U.shape[1]
    # Build tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                expected = 1.0
                err = abs(s - expected)
                if err > max_err:
                    max_err = err
    return max_err

def _verify_fast(U, V, W, n, m, p, tol=1e-10):
    """Faster verification using numpy einsum."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    # Reshape
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    max_err = np.max(np.abs(T - 1.0))
    return max_err

def _try_rank_r(n, m, p, r, deadline, rng, max_iters=200):
    """Try to find a rank-r decomposition via iterative refinement."""
    # Initialize with random values
    U = rng.standard_normal((n*m, r))
    V = rng.standard_normal((m*p, r))
    W = rng.standard_normal((p*n, r))
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    best_err = float('inf')
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute current tensor
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        err = np.max(np.abs(T - 1.0))
        if err < best_err:
            best_err = err
            if err < 1e-10:
                return U.copy(), V.copy(), W.copy(), err
        
        # Update U: for each (i,j), U3[i,j,:] should satisfy
        # sum_c U3[i,j,c] * (V3[j,k,c] * W3[k,i,c])_summed_over_k = 1
        # For fixed (i,j): U3[i,j,:] @ (V3[j,:,c]*W3[:,i,c]).sum(axis=0) = 1
        # Let M_U[i,j,c] = sum_k V3[j,k,c] * W3[k,i,c]
        # Then U3[i,j,:] @ M_U[i,j,:] = 1 for all (i,j)
        
        # M_U[i,j,c] = sum_k V3[j,k,c] * W3[k,i,c]
        M_U = np.einsum('jkc,kic->ijc', V3, W3)
        # For each (i,j), solve: U3[i,j,:] @ M_U[i,j,:] = 1
        # This is underdetermined if r > 1; use least squares or pseudoinverse
        # Actually we want U3[i,j,:] such that dot(U3[i,j,:], M_U[i,j,:]) = 1
        # Simplest: U3[i,j,:] = M_U[i,j,:] / (M_U[i,j,:] @ M_U[i,j,:]) if norm > eps
        
        for i in range(n):
            for j in range(m):
                mv = M_U[i, j, :]
                norm_sq = np.dot(mv, mv)
                if norm_sq > 1e-15:
                    U3[i, j, :] = mv / norm_sq
                else:
                    # Random restart for this row
                    U3[i, j, :] = rng.standard_normal(r)
        
        # Similarly update V
        # For fixed (j,k): V3[j,k,:] @ sum_i U3[i,j,c] * W3[k,i,c] = 1
        M_V = np.einsum('ijc,kic->jkc', U3, W3)
        for j in range(m):
            for k in range(p):
                mv = M_V[j, k, :]
                norm_sq = np.dot(mv, mv)
                if norm_sq > 1e-15:
                    V3[j, k, :] = mv / norm_sq
                else:
                    V3[j, k, :] = rng.standard_normal(r)
        
        # Similarly update W
        # For fixed (k,i): W3[k,i,:] @ sum_j U3[i,j,c] * V3[j,k,c] = 1
        M_W = np.einsum('ijc,jkc->kic', U3, V3)
        for k in range(p):
            for i in range(n):
                mv = M_W[k, i, :]
                norm_sq = np.dot(mv, mv)
                if norm_sq > 1e-15:
                    W3[k, i, :] = mv / norm_sq
                else:
                    W3[k, i, :] = rng.standard_normal(r)
        
        # Recompute error
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        err = np.max(np.abs(T - 1.0))
        if err < best_err:
            best_err = err
            if err < 1e-10:
                return U3.reshape(n*m, r).copy(), V3.reshape(m*p, r).copy(), W3.reshape(p*n, r).copy(), err
    
    return None, None, None, best_err

def _try_structured(n, m, p, deadline, rng):
    """Try structured decompositions based on known results."""
    # For 3x4x5, try ranks from lower bound up
    lower = max(n*m, m*p, p*n)  # 20
    upper = n*m*p  # 60
    
    # First, try to see if a lower rank is possible via SVD-based approach
    # Build the "matmul tensor" and try to find low-rank CP decomposition
    
    # Create the full tensor
    T = np.ones((n, m, p))
    
    # Try various ranks
    for r in range(lower, upper + 1):
        if time.time() > deadline:
            break
        result = _try_rank_r(n, m, p, r, deadline, rng, max_iters=100)
        if result[0] is not None:
            U, V, W, err = result
            if err < 1e-9:
                return U, V, W, r
    
    return None, None, None, upper

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # Start with schoolbook as fallback
    U_fallback, V_fallback, W_fallback = _schoolbook(n, m, p)
    
    rng = np.random.RandomState(42)
    
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Try to find lower rank
    lower = max(n*m, m*p, p*n)
    
    # Try structured search for each rank
    for r in range(lower, best_rank):
        if time.time() > deadline - 5:
            break
        U, V, W, err = _try_rank_r(n, m, p, r, deadline, rng, max_iters=150)
        if U is not None and err < 1e-9:
            # Verify with exact check
            if _verify(U, V, W, n, m, p, tol=1e-10) < 1e-9:
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break  # Found the minimal rank for this search
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
