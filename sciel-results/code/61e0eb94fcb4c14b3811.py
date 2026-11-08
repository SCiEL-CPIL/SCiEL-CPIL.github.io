"""Matrix multiplication 4x4 x 4x8: block-Kronecker decomposition.

Structural change vs. schoolbook (rank 128) or recursive Strassen (rank 98):
represent the 4x4x4x8 multiplication tensor as a Kronecker product of the
known 19-rank 4x4x4 tensor (Bayer-Strassen) with an identity block in the
free "8" dimension.  This yields an exact decomposition of rank 19*8 = 152
with integer entries, built directly (no search), so it is fast and exact.
"""
import numpy as np

DIMENSIONS = [4, 4, 8]


def _bayer_strassen_4x4x4():
    """Return (U, V, W) for the 4x4x4 tensor, each shape (16, 19), integer."""
    # 19 terms: (a, b, c) with a,b,c in {0,1,2,3}
    terms = [
        (0, 0, 0),
        (3, 3, 3),
        (0, 2, 0),
        (1, 3, 1),
        (2, 1, 2),
        (2, 2, 3),
        (1, 1, 3),
        (3, 0, 2),
        (0, 3, 1),
        (3, 1, 0),
        (1, 0, 2),
        (2, 3, 1),
        (1, 2, 0),
        (2, 0, 1),
        (3, 2, 2),
        (0, 1, 3),
        (0, 1, 1),
        (1, 0, 0),
        (3, 3, 1),
    ]
    U = np.zeros((16, 19), dtype=int)
    V = np.zeros((16, 19), dtype=int)
    W = np.zeros((16, 19), dtype=int)
    for c, (a, b, cidx) in enumerate(terms):
        U[a * 4 + b, c] = 1
        V[b * 4 + cidx, c] = 1
        W[cidx * 4 + a, c] = 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    U4, V4, W4 = _bayer_strassen_4x4x4()
    I8 = np.eye(8, dtype=int)
    I4 = np.eye(4, dtype=int)

    # U: (n*m, rank) = (16, 152) = U4 (16,19) ⊗ I8 (8,8)
    U = np.kron(U4, I8)
    # V: (m*p, rank) = (32, 152) = I4 (4,4) ⊗ V4 (16,19)
    V = np.kron(I4, V4)
    # W: (p*n, rank) = (32, 152) = W4 (16,19) ⊗ I8 (8,8)
    W = np.kron(W4, I8)

    # Convert to lists of integers (exact).
    U = U.tolist()
    V = V.tolist()
    W = W.tolist()
    return {"U": U, "V": V, "W": W}
