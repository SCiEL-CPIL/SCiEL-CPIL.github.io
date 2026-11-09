"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction
from collections import defaultdict

import numpy as np
from scipy import linalg

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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


def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _tensor_from_decomp(U, V, W):
    """Compute tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] using einsum."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    return np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)


def _verify_decomposition(decomp, T, tol=1e-9):
    """Check if the decomposition exactly reproduces T."""
    U, V, W = decomp['U'], decomp['V'], decomp['W']
    T_approx = _tensor_from_decomp(U, V, W)
    return np.allclose(T_approx, T, atol=tol, rtol=0)


def _snap_to_half_integers(decomp, tol=1e-6):
    """Snap entries to nearest half-integer if close."""
    def snap_matrix(M):
        M_snap = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                nearest = np.round(val * 2) / 2.0
                if abs(val - nearest) < tol:
                    M_snap[i, j] = nearest
                else:
                    M_snap[i, j] = val
        return M_snap

    U = np.array(decomp['U'], dtype=np.float64)
    V = np.array(decomp['V'], dtype=np.float64)
    W = np.array(decomp['W'], dtype=np.float64)
    
    U_snap = snap_matrix(U)
    V_snap = snap_matrix(V)
    W_snap = snap_matrix(W)
    
    return {
        'U': U_snap.tolist(),
        'V': V_snap.tolist(),
        'W': W_snap.tolist()
    }


def _try_find_smaller_decomposition(n, m, p, deadline):
    """
    Attempt to find a smaller decomposition by exploiting tensor structure.
    For 3x4x8, the schoolbook rank is 96.
    We try to find a rank r < 96 decomposition.
    
    Strategy:
    1. The tensor T has non-zero entries only at (i*m+j, j*p+k, k*n+i) for i,j,k.
    2. We can formulate this as a linear algebra problem.
    3. However, a direct search for minimal rank is NP-hard.
    4. We will try a heuristic based on grouping terms.
    
    For now, we return the schoolbook decomposition as a safe fallback.
    If we had time, we could try specific algebraic identities or
    optimization-based search.
    """
    # Check time
    if time.time() > deadline - 10:
        return n * m * p, _schoolbook(n, m, p)
    
    # For 3x4x8, let's see if we can do better than 96.
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(12, 32, 24) = 32.
    # But matrix multiplication tensors are known to have higher rank.
    # For 2x2x2, rank is 7 (Strassen) vs 8 schoolbook.
    # For 3x3x3, rank is 23 (Strassen-like) vs 27 schoolbook.
    # For 3x4x8, we don't have a known optimal rank, but likely < 96.
    
    # Let's try a simple optimization approach if we have time.
    # But given the 120s limit and complexity, we stick to schoolbook for safety.
    # A more sophisticated search would be needed.
    
    return n * m * p, _schoolbook(n, m, p)


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 40s buffer for the 150s limit
    
    # Try to find a smaller decomposition
    rank, decomp = _try_find_smaller_decomposition(n, m, p, deadline)
    
    # Verify the decomposition to ensure it is correct
    T = _build_target_tensor(n, m, p)
    if not _verify_decomposition(decomp, T):
        # Fallback to schoolbook if verification fails (should not happen)
        decomp = _schoolbook(n, m, p)
    
    # Try to snap to half-integers for cleaner output
    try:
        decomp = _snap_to_half_integers(decomp)
        if not _verify_decomposition(decomp, T):
            decomp = _schoolbook(n, m, p)
    except Exception:
        pass
    
    return decomp
# EVOLVE-BLOCK-END
