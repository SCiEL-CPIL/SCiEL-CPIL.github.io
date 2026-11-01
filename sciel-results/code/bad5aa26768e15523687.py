"""Rank-minimal exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Strategy: Start from rank-42 block-Strassen construction, then attempt to
reduce rank by identifying and eliminating linearly dependent tensor slices
using rational linear algebra. If rank-40 or rank-41 is feasible, return it;
otherwise return rank-42.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]


def _strassen_coefficients():
    """Return the 7 Strassen product specifications for 2x2x2."""
    products = [
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(1,0,1), (1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(0,1,1), (1,1,1)]},
        {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (1,0,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (0,1,1)]},
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]},
    ]
    return products


def _build_rank42_block():
    """Build rank-42 decomposition using block-partition + Strassen."""
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen = _strassen_coefficients()
    
    col = 0
    for r in range(3):
        for sub in range(2):
            a_offset = sub * 2
            b_row_offset = sub * 2
            
            for prod_idx in range(7):
                prod = strassen[prod_idx]
                for (i, j, coeff) in prod['U']:
                    U[i * m + (a_offset + j), col] += coeff
                for (j, kb, coeff) in prod['V']:
                    V[(b_row_offset + j) * p + (2 * r + kb), col] += coeff
                for (i, kb, coeff) in prod['W']:
                    W[(2 * r + kb) * n + i, col] += coeff
                col += 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                if abs(val - 1.0) > 1e-9:
                    return False, (i, j, k, val)
    return True, None


def _to_output(mat):
    """Convert numpy array to list of lists with int where possible."""
    return [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in mat]


def _build_schoolbook(n, m, p):
    """Build schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[ii * m + jj][col] = 1
                V[jj * p + kk][col] = 1
                W[kk * n + ii][col] = 1
                col += 1
    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, target_rank, timeout):
    """Attempt to reduce rank by finding linear dependencies among columns.
    
    The tensor equation is: for all (i,j,k), sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = 1
    
    We can try to express some columns as linear combinations of others.
    This is equivalent to finding a rank-revealing decomposition.
    
    Approach: treat the problem as finding matrices U', V', W' of rank r such that
    the tensor equation holds. Start from rank-42 and try to eliminate columns.
    """
    start = time.time()
    U_orig = U.copy()
    V_orig = V.copy()
    W_orig = W.copy()
    orig_rank = U_orig.shape[1]
    
    # Try to find if any column can be expressed as a combination of others
    # by checking if removing it and solving for the remaining still works.
    # This is expensive, so we'll use a heuristic: check column rank of W.
    
    # W has shape (p*n, rank) = (12, 42). If columns are linearly dependent,
    # we might be able to reduce.
    
    # Compute the rank of W
    W_rank = np.linalg.matrix_rank(W_orig, tol=1e-10)
    
    if W_rank < orig_rank:
        # There are linear dependencies. Find a basis for the column space.
        # Use QR with column pivoting to find independent columns.
        Q, R, piv = np.linalg.qr(W_orig, mode='complete', pivoting=True)
        # The first W_rank columns (in permuted order) form a basis
        # We can try to re-express the decomposition with fewer terms.
        
        # This is complex; for now, just note that reduction might be possible
        # but we'll fall back to the original if we can't cleanly extract it.
        pass
    
    # Alternative approach: try random rank-reduction via projection
    # Pick a random (orig_rank - target_rank) dimensional subspace to project onto
    # and check if the tensor equation still holds.
    
    for attempt in range(50):
        if time.time() > timeout:
            break
        
        # Random projection: pick target_rank random columns and see if
        # the remaining information can be captured
        cols = np.random.choice(orig_rank, target_rank, replace=False)
        cols_sorted = np.sort(cols)
        
        U_sub = U_orig[:, cols_sorted]
        V_sub = V_orig[:, cols_sorted]
        W_sub = W_orig[:, cols_sorted]
        
        # Check if this sub-decomposition is valid
        valid, _ = _verify(U_sub, V_sub, W_sub, n, m, p)
        if valid:
            return U_sub, V_sub, W_sub
    
    return None, None, None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Build rank-42 block-partition + Strassen decomposition
    U, V, W = _build_rank42_block()

    # Verify correctness
    valid, info = _verify(U, V, W, n, m, p)
    if not valid:
        # Fallback: schoolbook rank 48
        U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Try to reduce rank below 42
    if time.time() < deadline - 30.0:
        # Try rank 41
        U41, V41, W41 = _try_reduce_rank(U, V, W, n, m, p, 41, deadline - 15.0)
        if U41 is not None:
            return {"U": _to_output(U41), "V": _to_output(V41), "W": _to_output(W41)}
        
        # Try rank 40
        if time.time() < deadline - 15.0:
            U40, V40, W40 = _try_reduce_rank(U, V, W, n, m, p, 40, deadline - 10.0)
            if U40 is not None:
                return {"U": _to_output(U40), "V": _to_output(V40), "W": _to_output(W40)}

    # Return rank-42
    return {"U": _to_output(U), "V": _to_output(V), "W": _to_output(W)}
