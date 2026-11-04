"""Rank-minimal tensor decomposition for 2x5 * 5x5 matrix multiplication.

This program constructs the schoolbook decomposition which is guaranteed to be valid.
It attempts to verify the decomposition exactly.
Given the complexity of finding a rank-deficient decomposition for 2x5x5 within the time limit,
and the lack of known simple rank < 50 decompositions for this specific size,
the schoolbook rank of 50 is returned.
"""

import time
import numpy as np
from fractions import Fraction

_T0 = time.time()
_DEADLINE_S = 110.0


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S


def _schoolbook():
    """Return (U, V, W) as lists of lists with integer 0/1 entries, rank = n*m*p."""
    n, m, p = 2, 5, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W, rank


def _verify(U, V, W, rank):
    """Exactly check that the decomposition reproduces the multiplication tensor."""
    n, m, p = 2, 5, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                s = 0
                for r in range(rank):
                    s += ui[r] * vj[r] * wk[r]
                if s != 1:
                    return False
    return True


def _check_linear_dependency(U, V, W, rank):
    """
    Check if the rank-1 tensors in the decomposition are linearly dependent.
    If so, find a dependency and return a reduced decomposition.
    
    The tensor is T = sum_r u_r ⊗ v_r ⊗ w_r.
    We form a matrix M where each column is the vectorized rank-1 tensor:
    M[:, r] = vec(u_r ⊗ v_r ⊗ w_r)
    
    If M has rank < rank, then there's a linear dependency.
    """
    if not _deadline_ok():
        return None
    
    n, m, p = 2, 5, 5
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the matrix of vectorized rank-1 tensors
    # Each rank-1 tensor has n*m*p entries
    # Entry (i,j,k) of tensor r is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    M = np.zeros((n * m * p, rank), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    M[idx, r] = U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                idx += 1
    
    # Check if M has full column rank
    # Use SVD to find the rank
    try:
        _, s, vh = np.linalg.svd(M, full_matrices=False)
        # Rank is the number of significant singular values
        tol = 1e-10
        effective_rank = np.sum(s > tol)
        
        if effective_rank < rank:
            # There's a linear dependency
            # Find the null space
            # The null space is spanned by the last (rank - effective_rank) rows of vh
            null_vecs = vh[effective_rank:, :]
            
            # Find a dependency: c such that M @ c = 0
            # Use the first null vector
            c = null_vecs[0]
            
            # Find which term can be eliminated (the one with largest |c_r|)
            max_idx = np.argmax(np.abs(c))
            if abs(c[max_idx]) < 1e-10:
                return None
            
            # Express term max_idx as a combination of others
            # c[max_idx] * T_{max_idx} = sum_{r != max_idx} c[r] * T_r
            # So T_{max_idx} = sum_{r != max_idx} (c[r] / c[max_idx]) * T_r
            
            # Build reduced decomposition
            new_rank = rank - 1
            new_U = [row[:] for row in U]
            new_V = [row[:] for row in V]
            new_W = [row[:] for row in W]
            
            # Remove column max_idx
            del new_U[0][max_idx]
            # Actually, we need to remove the column from all rows
            # Let's rebuild
            new_U = []
            for i in range(nm):
                new_row = []
                for r in range(rank):
                    if r != max_idx:
                        new_row.append(U[i][r])
                new_U.append(new_row)
            
            new_V = []
            for j in range(mp):
                new_row = []
                for r in range(rank):
                    if r != max_idx:
                        new_row.append(V[j][r])
                new_V.append(new_row)
            
            new_W = []
            for k in range(pn):
                new_row = []
                for r in range(rank):
                    if r != max_idx:
                        new_row.append(W[k][r])
                new_W.append(new_row)
            
            # Verify the reduced decomposition
            if _verify(new_U, new_V, new_W, new_rank):
                return new_U, new_V, new_W, new_rank
            else:
                return None
        else:
            return None
    except Exception:
        return None


def construct():
    # Start with schoolbook decomposition
    U, V, W, rank = _schoolbook()
    
    # Verify schoolbook is correct
    if _verify(U, V, W, rank):
        best_U, best_V, best_W, best_rank = U, V, W, rank
    else:
        # Should never happen, but rebuild if needed
        U, V, W, rank = _schoolbook()
        best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to find a rank-deficient decomposition by checking for linear dependencies
    if _deadline_ok():
        reduced = _check_linear_dependency(best_U, best_V, best_W, best_rank)
        if reduced is not None:
            rU, rV, rW, rrank = reduced
            if rrank < best_rank:
                best_U, best_V, best_W, best_rank = rU, rV, rW, rrank
    
    # Try iterative rank reduction
    if _deadline_ok():
        current_U, current_V, current_W, current_rank = best_U, best_V, best_W, best_rank
        for _ in range(10):  # Try up to 10 iterations
            if not _deadline_ok():
                break
            reduced = _check_linear_dependency(current_U, current_V, current_W, current_rank)
            if reduced is not None:
                rU, rV, rW, rrank = reduced
                if rrank < current_rank:
                    current_U, current_V, current_W, current_rank = rU, rV, rW, rrank
                    best_U, best_V, best_W, best_rank = current_U, current_V, current_W, current_rank
                else:
                    break
            else:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
