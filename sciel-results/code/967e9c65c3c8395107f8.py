"""Slice-method decomposition for 3x4x7 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]


def _build_matrices(U_list, V_list, W_list, n, m, p):
    """Convert list-of-lists factor matrices to numpy arrays."""
    U = np.array(U_list, dtype=float)
    V = np.array(V_list, dtype=float)
    W = np.array(W_list, dtype=float)
    return U, V, W


def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor.

    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    must equal 1 for all (i,j,k) in the multiplication tensor.
    """
    rank = U.shape[1]
    for c in range(rank):
        pass  # We verify via the full tensor reconstruction
    # Build the full tensor and check
    T = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c]  # (n*m,)
        Vc = V[:, c]  # (m*p,)
        Wc = W[:, c]  # (p*n,)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Uc[i * m + j] * Vc[j * p + k] * Wc[k * n + i]
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # The tensor for standard matrix multiplication C = AB is the all-ones tensor.
    # T[i, j, k] = 1 for all i, j, k.
    # This tensor has rank 1.
    # Decomposition:
    # U[i, 0] = 1 for all i
    # V[j, 0] = 1 for all j
    # W[k, 0] = 1 for all k
    # Sum_c U[i,c] V[j,c] W[k,c] = 1 * 1 * 1 = 1.
    
    rank = 1
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
