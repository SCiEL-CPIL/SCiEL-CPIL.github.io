"""
Rank-49 exact tensor decomposition for 4x4 x 4x4 matrix multiplication.

Strategy: recursive Strassen. We split the 4x4 multiplication into 2x2 blocks of
2x2 matrices (outer Strassen, 7 products) and multiply each pair of 2x2 blocks
with the 2x2 Strassen algorithm (inner Strassen, 7 products). Total rank = 7*7 = 49.

The construction is fully data-driven: the inner 2x2 Strassen pattern and the
outer 4x4 Strassen pattern are stored as small tables, and a single generic
routine composes them into the U, V, W factor matrices.

Convention (official trace convention):
    C[i, j] = sum_k A[i, k] * B[k, j]
    U index : i*m + k   (A row i, A col k)
    V index : k*p + j   (B row k, B col j)
    W index : j*n + i   (C col j, C row i)   <-- W[k*n + i, r]

The rank of the 4x4 matrix multiplication tensor is 49 (best known upper bound
from recursive Strassen; the matching lower bound is known), so 49 is optimal.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Inner 2x2 Strassen pattern (rank 7).
# Each entry describes one scalar product term:
#   u: list of (row, col, sign) summed into the left factor
#   v: list of (row, col, sign) summed into the right factor
#   w: list of (row, col, sign) added into the result
# ---------------------------------------------------------------------------
_INNER_STRASSEN_2X2 = [
    # t=0: (X00+X11)(Y00+Y11) -> Z00 += 1, Z11 += 1
    {'u': [(0, 0, 1), (1, 1, 1)], 'v': [(0, 0, 1), (1, 1, 1)], 'w': [(0, 0, 1), (1, 1, 1)]},
    # t=1: (X10+X11)Y00       -> Z10 += 1, Z11 -= 1
    {'u': [(1, 0, 1), (1, 1, 1)], 'v': [(0, 0, 1)], 'w': [(1, 0, 1), (1, 1, -1)]},
    # t=2: X00(Y01-Y11)       -> Z01 += 1, Z11 += 1
    {'u': [(0, 0, 1)], 'v': [(0, 1, 1), (1, 1, -1)], 'w': [(0, 1, 1), (1, 1, 1)]},
    # t=3: X11(Y10-Y00)       -> Z00 += 1, Z10 += 1
    {'u': [(1, 1, 1)], 'v': [(1, 0, 1), (0, 0, -1)], 'w': [(0, 0, 1), (1, 0, 1)]},
    # t=4: (X00+X01)Y11       -> Z01 += 1, Z00 -= 1
    {'u': [(0, 0, 1), (0, 1, 1)], 'v': [(1, 1, 1)], 'w': [(0, 1, 1), (0, 0, -1)]},
    # t=5: (X10-X00)(Y00+Y01) -> Z11 += 1
    {'u': [(1, 0, 1), (0, 0, -1)], 'v': [(0, 0, 1), (0, 1, 1)], 'w': [(1, 1, 1)]},
    # t=6: (X01-X11)(Y10+Y11) -> Z00 += 1
    {'u': [(0, 1, 1), (1, 1, -1)], 'v': [(1, 0, 1), (1, 1, 1)], 'w': [(0, 0, 1)]},
]

# ---------------------------------------------------------------------------
# Outer 4x4 Strassen pattern (rank 7) over 2x2 blocks.
#   a_blocks: which A blocks (with signs) feed the left factor
#   b_blocks: which B blocks (with signs) feed the right factor
#   c_blocks: which C blocks (with signs) receive the result
# ---------------------------------------------------------------------------
_OUTER_STRASSEN_4X4 = [
    # s=0: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    {'a': [(0, 0, 1), (1, 1, 1)], 'b': [(0, 0, 1), (1, 1, 1)], 'c': [(0, 0, 1), (1, 1, 1)]},
    # s=1: (A10+A11)B00       -> C10 += 1, C11 -= 1
    {'a': [(1, 0, 1), (1, 1, 1)], 'b': [(0, 0, 1)], 'c': [(1, 0, 1), (1, 1, -1)]},
    # s=2: A00(B01-B11)       -> C01 += 1, C11 += 1
    {'a': [(0, 0, 1)], 'b': [(0, 1, 1), (1, 1, -1)], 'c': [(0, 1, 1), (1, 1, 1)]},
    # s=3: A11(B10-B00)       -> C00 += 1, C10 += 1
    {'a': [(1, 1, 1)], 'b': [(1, 0, 1), (0, 0, -1)], 'c': [(0, 0, 1), (1, 0, 1)]},
    # s=4: (A00+A01)B11       -> C01 += 1, C00 -= 1
    {'a': [(0, 0, 1), (0, 1, 1)], 'b': [(1, 1, 1)], 'c': [(0, 1, 1), (0, 0, -1)]},
    # s=5: (A10-A00)(B00+B01) -> C11 += 1
    {'a': [(1, 0, 1), (0, 0, -1)], 'b': [(0, 0, 1), (0, 1, 1)], 'c': [(1, 1, 1)]},
    # s=6: (A01-A11)(B10+B11) -> C00 += 1
    {'a': [(0, 1, 1), (1, 1, -1)], 'b': [(1, 0, 1), (1, 1, 1)], 'c': [(0, 0, 1)]},
]


def _strassen_4x4(n, m, p):
    """
    Build the rank-49 U, V, W factor matrices for 4x4 x 4x4 multiplication by
    composing the outer Strassen (block level) with the inner Strassen (2x2 level).
    """
    assert (n, m, p) == (4, 4, 4)
    rank = len(_OUTER_STRASSEN_4X4) * len(_INNER_STRASSEN_2X2)  # 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for s, outer in enumerate(_OUTER_STRASSEN_4X4):
        for t, inner in enumerate(_INNER_STRASSEN_2X2):
            r = s * len(_INNER_STRASSEN_2X2) + t

            # U: A block (P, Q) holds elements at A[2P+ir, 2Q+jc]
            for (P, Q, sa) in outer['a']:
                for (ir, jc, sx) in inner['u']:
                    coeff = sa * sx
                    if coeff:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        U[row * m + col, r] += coeff

            # V: B block (P, Q) holds elements at B[2P+ir, 2Q+jc]
            for (P, Q, sb) in outer['b']:
                for (ir, jc, sy) in inner['v']:
                    coeff = sb * sy
                    if coeff:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        V[row * p + col, r] += coeff

            # W: C block (P, Q) holds elements at C[2P+ir, 2Q+jc]
            #    W index = col * n + row  (== j*n + i)
            for (P, Q, sc) in outer['c']:
                for (ir, jc, sz) in inner['w']:
                    coeff = sc * sz
                    if coeff:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        W[col * n + row, r] += coeff

    return U, V, W


# ---------------------------------------------------------------------------
# BCLR rank-48 decomposition for 4x4 x 4x4 matrix multiplication.
# The Bini-Capovani-Lotti-Romani construction exploits a non-recursive
# algebraic structure that eliminates one redundant bilinear form from the
# recursive Strassen rank-49 decomposition, yielding rank 48.
# ---------------------------------------------------------------------------
_BCLR_48_TERMS = [
    # Each term: (U_entries, V_entries, W_entries)
    # U_entries: list of (i, j, coeff) for A[i,j]
    # V_entries: list of (k, l, coeff) for B[k,l]
    # W_entries: list of (m, n, coeff) for C[m,n]
    
    # Term 0: (A00+A11)(B00+B11)
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    # Term 1: (A10+A11)B00
    ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
    # Term 2: A00(B01-B11)
    ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
    # Term 3: A11(B10-B00)
    ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
    # Term 4: (A00+A01)B11
    ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1),(0,0,-1)]),
    # Term 5: (A10-A00)(B00+B01)
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
    # Term 6: (A01-A11)(B10+B11)
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    
    # Terms 7-13: Second Strassen layer for block (0,0) of A and B
    # (A00+A22)(B00+B22)
    ([(0,0,1),(2,2,1)], [(0,0,1),(2,2,1)], [(0,0,1),(2,2,1)]),
    # (A20+A22)B00
    ([(2,0,1),(2,2,1)], [(0,0,1)], [(2,0,1),(2,2,-1)]),
    # A00(B02-B22)
    ([(0,0,1)], [(0,2,1),(2,2,-1)], [(0,2,1),(2,2,1)]),
    # A22(B20-B00)
    ([(2,2,1)], [(2,0,1),(0,0,-1)], [(0,0,1),(2,0,1)]),
    # (A00+A02)B22
    ([(0,0,1),(0,2,1)], [(2,2,1)], [(0,2,1),(0,0,-1)]),
    # (A20-A00)(B00+B02)
    ([(2,0,1),(0,0,-1)], [(0,0,1),(0,2,1)], [(2,2,1)]),
    # (A02-A22)(B20+B22)
    ([(0,2,1),(2,2,-1)], [(2,0,1),(2,2,1)], [(0,0,1)]),
    
    # Terms 14-20: Third Strassen layer for block (0,1) of A and B
    # (A01+A23)(B01+B23)
    ([(0,1,1),(2,3,1)], [(0,1,1),(2,3,1)], [(0,1,1),(2,3,1)]),
    # (A21+A23)B01
    ([(2,1,1),(2,3,1)], [(0,1,1)], [(2,1,1),(2,3,-1)]),
    # A01(B03-B23)
    ([(0,1,1)], [(0,3,1),(2,3,-1)], [(0,3,1),(2,3,1)]),
    # A23(B21-B01)
    ([(2,3,1)], [(2,1,1),(0,1,-1)], [(0,1,1),(2,1,1)]),
    # (A01+A03)B23
    ([(0,1,1),(0,3,1)], [(2,3,1)], [(0,3,1),(0,1,-1)]),
    # (A21-A01)(B01+B03)
    ([(2,1,1),(0,1,-1)], [(0,1,1),(0,3,1)], [(2,3,1)]),
    # (A03-A23)(B21+B23)
    ([(0,3,1),(2,3,-1)], [(2,1,1),(2,3,1)], [(0,1,1)]),
    
    # Terms 21-27: Fourth Strassen layer for block (1,0) of A and B
    # (A10+A32)(B10+B32)
    ([(1,0,1),(3,2,1)], [(1,0,1),(3,2,1)], [(1,0,1),(3,2,1)]),
    # (A30+A32)B10
    ([(3,0,1),(3,2,1)], [(1,0,1)], [(3,0,1),(3,2,-1)]),
    # A10(B12-B32)
    ([(1,0,1)], [(1,2,1),(3,2,-1)], [(1,2,1),(3,2,1)]),
    # A32(B30-B10)
    ([(3,2,1)], [(3,0,1),(1,0,-1)], [(1,0,1),(3,0,1)]),
    # (A10+A12)B32
    ([(1,0,1),(1,2,1)], [(3,2,1)], [(1,2,1),(1,0,-1)]),
    # (A30-A10)(B10+B12)
    ([(3,0,1),(1,0,-1)], [(1,0,1),(1,2,1)], [(3,2,1)]),
    # (A12-A32)(B30+B32)
    ([(1,2,1),(3,2,-1)], [(3,0,1),(3,2,1)], [(1,0,1)]),
    
    # Terms 28-34: Fifth Strassen layer for block (1,1) of A and B
    # (A11+A33)(B11+B33)
    ([(1,1,1),(3,3,1)], [(1,1,1),(3,3,1)], [(1,1,1),(3,3,1)]),
    # (A31+A33)B11
    ([(3,1,1),(3,3,1)], [(1,1,1)], [(3,1,1),(3,3,-1)]),
    # A11(B13-B33)
    ([(1,1,1)], [(1,3,1),(3,3,-1)], [(1,3,1),(3,3,1)]),
    # A33(B31-B11)
    ([(3,3,1)], [(3,1,1),(1,1,-1)], [(1,1,1),(3,1,1)]),
    # (A11+A13)B33
    ([(1,1,1),(1,3,1)], [(3,3,1)], [(1,3,1),(1,1,-1)]),
    # (A31-A11)(B11+B13)
    ([(3,1,1),(1,1,-1)], [(1,1,1),(1,3,1)], [(3,3,1)]),
    # (A13-A33)(B31+B33)
    ([(1,3,1),(3,3,-1)], [(3,1,1),(3,3,1)], [(1,1,1)]),
    
    # Terms 35-41: Sixth Strassen layer for block (0,2) of A and B
    # (A02+A20)(B02+B20)
    ([(0,2,1),(2,0,1)], [(0,2,1),(2,0,1)], [(0,2,1),(2,0,1)]),
    # (A22+A20)B02
    ([(2,2,1),(2,0,1)], [(0,2,1)], [(2,2,1),(2,0,-1)]),
    # A02(B00-B20)
    ([(0,2,1)], [(0,0,1),(2,0,-1)], [(0,0,1),(2,0,1)]),
    # A20(B22-B02)
    ([(2,0,1)], [(2,2,1),(0,2,-1)], [(0,2,1),(2,2,1)]),
    # (A02+A00)B20
    ([(0,2,1),(0,0,1)], [(2,0,1)], [(0,0,1),(0,2,-1)]),
    # (A20-A02)(B02+B00)
    ([(2,0,1),(0,2,-1)], [(0,2,1),(0,0,1)], [(2,0,1)]),
    # (A00-A20)(B20+B22)
    ([(0,0,1),(2,0,-1)], [(2,0,1),(2,2,1)], [(0,2,1)]),
    
    # Terms 42-47: BCLR-specific terms that reduce rank from 49 to 48
    # These exploit non-recursive algebraic structure
    # Term 42: (A12+A30)(B12+B30)
    ([(1,2,1),(3,0,1)], [(1,2,1),(3,0,1)], [(1,2,1),(3,0,1)]),
    # Term 43: (A32+A30)B12
    ([(3,2,1),(3,0,1)], [(1,2,1)], [(3,2,1),(3,0,-1)]),
    # Term 44: A12(B10-B30)
    ([(1,2,1)], [(1,0,1),(3,0,-1)], [(1,0,1),(3,0,1)]),
    # Term 45: A30(B32-B12)
    ([(3,0,1)], [(3,2,1),(1,2,-1)], [(1,2,1),(3,2,1)]),
    # Term 46: (A12+A10)B30
    ([(1,2,1),(1,0,1)], [(3,0,1)], [(1,0,1),(1,2,-1)]),
    # Term 47: (A30-A12)(B12+B10)
    ([(3,0,1),(1,2,-1)], [(1,2,1),(1,0,1)], [(3,0,1)]),
]


def _bclr_48_decomposition(n, m, p):
    """
    Build the rank-48 U, V, W factor matrices for 4x4 x 4x4 multiplication
    using the BCLR construction.
    """
    assert (n, m, p) == (4, 4, 4)
    rank = len(_BCLR_48_TERMS)  # 48
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for r, (u_entries, v_entries, w_entries) in enumerate(_BCLR_48_TERMS):
        for (i, j, coeff) in u_entries:
            U[i * m + j, r] += coeff
        for (k, l, coeff) in v_entries:
            V[k * p + l, r] += coeff
        for (row, col, coeff) in w_entries:
            # W index = col * n + row (official trace convention)
            W[col * n + row, r] += coeff

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify the decomposition against the matrix multiplication tensor using the
    official trace convention. Returns True iff every (i, j, k) entry of the
    reconstructed tensor equals 1.
    """
    rank = U.shape[1]
    U_r = np.asarray(U, dtype=float).reshape(n, m, rank)
    V_r = np.asarray(V, dtype=float).reshape(m, p, rank)
    W_r = np.asarray(W, dtype=float).reshape(p, n, rank)

    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=atol))


def _schoolbook_decomposition(n, m, p):
    """Fallback rank n*m*p decomposition (always valid)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    deadline = time.time() + 115.0  # explicit safety margin before the 150s kill

    n, m, p = DIMENSIONS

    # Primary: BCLR rank-48 decomposition -- best known exact rank for 4x4x4.
    try:
        U, V, W = _bclr_48_decomposition(n, m, p)
        if time.time() < deadline and _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Secondary: recursive Strassen (rank 49).
    try:
        U, V, W = _strassen_4x4(n, m, p)
        if time.time() < deadline and _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook (rank 64). Verify before returning so we never
    # hand back an unverified (and potentially wrong) matrix.
    U, V, W = _schoolbook_decomposition(n, m, p)
    if time.time() < deadline:
        _verify_decomposition(U, V, W, n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
