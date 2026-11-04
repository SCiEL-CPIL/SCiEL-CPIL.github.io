"""Task: rank-minimal exact tensor decomposition of 3x4 by 4x7 matrix multiplication.
Trace convention: out[i,k] = sum_j A[i,j]*B[j,k] == sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].

Strategy: Constructive rank-21 decomposition. For each (i,k) pair, create one
rank-1 term where U is nonzero on all j for that i, V is nonzero on all j for
that k, and W is a single indicator. This exploits the all-ones structure of
the matmul tensor to share across the middle dimension j.

We then attempt to reduce the rank via elimination using a time-bounded search.
"""
import numpy as np
import time
import itertools
import random

_START = time.time()
_LIMIT = 130.0
DIMENSIONS = [3, 4, 7]


def _elapsed():
    return time.time() - _START


def _remaining():
    return _LIMIT - _elapsed()


def _check(U, V, W, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] equals the matmul tensor.
    
    Returns (max_error, n_nonzero) where max_error is the max absolute error
    and n_nonzero is the number of non-zero tensor entries.
    """
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n
    
    R = U.shape[1]
    # Build tensor from decomposition
    T_decomp = np.zeros((Nm, Mp, Pn))
    for r in range(R):
        T_decomp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    # Build the true matmul tensor
    T_true = np.zeros((Nm, Mp, Pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i * m + j, j * p + k, k * n + i] = 1.0
    
    diff = T_decomp - T_true
    max_err = np.max(np.abs(diff))
    n_nonzero = np.sum(np.abs(diff) > tol)
    
    return max_err, n_nonzero


def _construct_base():
    """Construct the base rank-21 decomposition."""
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n

    R = n * p  # 3 * 7 = 21
    U = np.zeros((Nm, R))
    V = np.zeros((Mp, R))
    W = np.zeros((Pn, R))
    
    col = 0
    for i in range(n):
        for k in range(p):
            # U: nonzero for all j when row index is i*m+j
            for j in range(m):
                U[i * m + j, col] = 1.0
            # V: nonzero on all j when row index is j*p+k
            for j in range(m):
                V[j * p + k, col] = 1.0
            # W: single indicator at k*n+i
            W[k * n + i, col] = 1.0
            col += 1
    
    return U, V, W


def _try_reduce_rank(U, V, W, max_rank_reduction=3, timeout=30.0):
    """Attempt to reduce the rank of the decomposition by eliminating terms.
    
    Uses a greedy approach: for each term, try to express it as a linear combination
    of the other terms. If possible, eliminate it.
    
    Returns (U_reduced, V_reduced, W_reduced, rank_reduced)
    """
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n
    R = U.shape[1]
    
    # Build the tensor
    T = np.zeros((Nm, Mp, Pn))
    for r in range(R):
        T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    # Try to eliminate terms one by one
    active = list(range(R))
    max_attempts = max_rank_reduction
    
    for _ in range(max_attempts):
        if _remaining() < timeout:
            break
            
        eliminated = False
        # Try each active term
        for idx, r in enumerate(active):
            if len(active) <= 2:
                break
                
            # Try to express term r as linear combination of other active terms
            others = [a for a in active if a != r]
            
            # Build the linear system
            # We want: U[:,r] * V[:,r] * W[:,r] = sum_{s in others} c_s * U[:,s] * V[:,s] * W[:,s]
            # This is a linear system in c_s
            # The tensor T_r = U[:,r] * V[:,r] * W[:,r] should be in the span of {T_s : s in others}
            
            T_r = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            # Build matrix where each column is T_s flattened
            M = np.zeros((Nm * Mp * Pn, len(others)))
            for j, s in enumerate(others):
                M[:, j] = np.einsum('i,j,k->ijk', U[:, s], V[:, s], W[:, s]).flatten()
            
            # Solve least squares
            T_r_flat = T_r.flatten()
            
            # Check if T_r is in the span
            # Use SVD for numerical stability
            try:
                # If M has full column rank, use lstsq
                # Otherwise, check if T_r is in the column space
                U_svd, svals, Vt_svd = np.linalg.svd(M, full_matrices=False)
                
                # Project T_r onto the column space
                proj = U_svd[:, :len(others)] @ (U_svd[:, :len(others)].T @ T_r_flat)
                
                residual = np.linalg.norm(T_r_flat - proj)
                norm_Tr = np.linalg.norm(T_r_flat)
                
                if norm_Tr > 1e-12 and residual / norm_Tr < 1e-8:
                    # T_r is in the span, eliminate it
                    active.remove(r)
                    eliminated = True
                    break
            except np.linalg.LinAlgError:
                continue
        
        if not eliminated:
            break
    
    # Reconstruct with only active terms
    R_new = len(active)
    U_new = U[:, active]
    V_new = V[:, active]
    W_new = W[:, active]
    
    return U_new, V_new, W_new, R_new


def construct():
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n

    # Start with base rank-21 decomposition
    U, V, W = _construct_base()
    
    # Verify the decomposition
    max_err, n_bad = _check(U, V, W)
    
    if max_err > 1e-6:
        # Fall back to schoolbook if something went wrong
        R0 = n * m * p
        U0 = np.zeros((Nm, R0))
        V0 = np.zeros((Mp, R0))
        W0 = np.zeros((Pn, R0))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i * m + j, col] = 1.0
                    V0[j * p + k, col] = 1.0
                    W0[k * n + i, col] = 1.0
                    col += 1
        U, V, W = U0, V0, W0
    else:
        # Try to reduce the rank
        if _remaining() > 10:
            U, V, W, R_new = _try_reduce_rank(U, V, W, max_rank_reduction=3, timeout=25.0)
            # Verify the reduced decomposition
            max_err_new, n_bad_new = _check(U, V, W)
            if max_err_new > 1e-6:
                # Revert if verification fails
                U, V, W = _construct_base()
    
    def _pack(arr):
        return [[float(x) for x in row] for row in np.asarray(arr)]
    
    return {"U": _pack(U), "V": _pack(V), "W": _pack(W)}
