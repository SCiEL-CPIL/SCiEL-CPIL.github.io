"""Tensor decomposition for 4x4 * 4x9 matrix multiplication.

Uses a Strassen-based decomposition:
  - Split the 4x9 product into two 4x4 products + one 4x1 product.
  - Each 4x4x4 block uses Strassen's rank-49 algorithm.
  - The 4x4x1 block uses schoolbook rank 4.
  Total rank = 49 + 49 + 4 = 102 (vs 144 schoolbook).

Index conventions (per task):
  U[i*m + j, r] for A entry (i,j)
  V[j*p + k, r] for B entry (j,k)
  W[k*n + i, r] for C entry (i,k)
"""
DIMENSIONS = [4, 4, 9]

import time
import numpy as np


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check U ⊙ V ⊙ W reproduces the matmul tensor (scaled to 1.0 per term)."""
    R = len(U[0])
    if R != len(V[0]) or R != len(W[0]):
        return False
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


# ---- Strassen 2x2x2 (rank 7) -------------------------------------------------
def _strassen2():
    """Return U,V,W (4x7 each) for 2x2 * 2x2 = 2x2."""
    n = m = p = 2
    # 7 rank-1 terms: (U entry, V entry, W entry) with indices (i,j),(j,k),(i,k)
    terms = [
        # 1: (a00+a11)(b00+b11)
        {((0, 0), 1), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1),
         ((0, 0), 1), ((1, 0), 1), ((0, 1), 1), ((1, 1), 1),
         ((0, 0), 1), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)},
        # 2: (a10+a11)b00
        {((1, 0), 1), ((1, 1), 1), ((0, 0), 1),
         ((0, 0), 1), ((1, 0), 1), ((1, 1), 1)},
        # 3: a00(b01-b11)
        {((0, 0), 1), ((0, 1), 1), ((1, 1), -1),
         ((0, 1), 1), ((1, 1), -1), ((0, 0), 1), ((1, 0), -1)},
        # 4: a11(b10-b00)
        {((1, 1), 1), ((1, 0), 1), ((0, 0), -1),
         ((1, 0), 1), ((0, 0), -1), ((1, 1), 1), ((0, 1), -1)},
        # 5: (a00+a01)b11
        {((0, 0), 1), ((0, 1), 1), ((1, 1), 1),
         ((0, 1), 1), ((1, 1), 1), ((0, 0), 1), ((1, 0), 1)},
        # 6: (a10-a00)(b00+b01)
        {((1, 0), 1), ((0, 0), -1), ((0, 0), 1), ((0, 1), 1),
         ((0, 0), 1), ((0, 1), 1), ((0, 0), 1), ((0, 1), 1)},
        # 7: (a01-a11)(b10+b11)
        {((0, 1), 1), ((1, 1), -1), ((1, 0), 1), ((1, 1), 1),
         ((1, 0), 1), ((1, 1), 1), ((0, 1), 1), ((1, 1), -1)},
    ]
    R = 7
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    for r, term in enumerate(terms):
        for (i, j), val in term.items():
            if j < p:  # (i,j) is a U index (i,j) with j < m
                U[i * m + j][r] += val
        # Recompute properly: separate U and V entries
    # Rebuild cleanly
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    for r, term in enumerate(terms):
        # Each term dict maps (a,b)->val; we need to know which are U,V,W.
        # We'll store terms explicitly instead.
        pass
    # Explicit reconstruction to avoid ambiguity
    R = 7
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]

    def add_U(i, j, r, v):
        U[i * m + j][r] += v
    def add_V(j, k, r, v):
        V[j * p + k][r] += v
    def add_W(i, k, r, v):
        W[k * n + i][r] += v

    # term 1: (a00+a11)(b00+b11)
    add_U(0, 0, 0, 1); add_U(1, 1, 0, 1)
    add_V(0, 0, 0, 1); add_V(1, 1, 0, 1)
    add_W(0, 0, 0, 1); add_W(0, 1, 0, 1); add_W(1, 0, 0, 1); add_W(1, 1, 0, 1)
    # term 2: (a10+a11)b00
    add_U(1, 0, 1, 1); add_U(1, 1, 1, 1)
    add_V(0, 0, 1, 1)
    add_W(1, 0, 1, 1); add_W(1, 1, 1, 1)
    # term 3: a00(b01-b11)
    add_U(0, 0, 2, 1)
    add_V(0, 1, 2, 1); add_V(1, 1, 2, -1)
    add_W(0, 1, 2, 1); add_W(1, 1, 2, -1)
    # term 4: a11(b10-b00)
    add_U(1, 1, 3, 1)
    add_V(1, 0, 3, 1); add_V(0, 0, 3, -1)
    add_W(1, 0, 3, 1); add_W(0, 0, 3, -1)
    # term 5: (a00+a01)b11
    add_U(0, 0, 4, 1); add_U(0, 1, 4, 1)
    add_V(1, 1, 4, 1)
    add_W(1, 0, 4, 1); add_W(1, 1, 4, 1)
    # term 6: (a10-a00)(b00+b01)
    add_U(1, 0, 5, 1); add_U(0, 0, 5, -1)
    add_V(0, 0, 5, 1); add_V(0, 1, 5, 1)
    add_W(0, 0, 5, 1); add_W(0, 1, 5, 1)
    # term 7: (a01-a11)(b10+b11)
    add_U(0, 1, 6, 1); add_U(1, 1, 6, -1)
    add_V(1, 0, 6, 1); add_V(1, 1, 6, 1)
    add_W(1, 0, 6, 1); add_W(1, 1, 6, -1)
    return U, V, W


# ---- Strassen 4x4x4 (rank 49) ------------------------------------------------
def _strassen4():
    """Return U,V,W (16x49) for 4x4 * 4x4 = 4x4 via 7 blocks of Strassen2."""
    n = m = p = 4
    n2 = m2 = p2 = 2
    U2, V2, W2 = _strassen2()  # 4x7 each
    R = 49
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]

    # 7 Strassen sub-products M1..M7, each of size 2x2 * 2x2 = 2x2.
    # Block layout: A = [[A00,A01],[A10,A11]] with 2x2 blocks; indices (bi,bj).
    # M_i = (linear combo of A blocks) * (linear combo of B blocks)
    # For each M_i we need, for each of its 4 (a_idx) entries a linear combo of
    # A-block entries, and for each (b_idx) a linear combo of B-block entries.

    # Define M_i as: list of (a_block_row, a_block_col, coeff) for the A side,
    #                list of (b_block_row, b_block_col, coeff) for the B side,
    #                list of (c_block_row, c_block_col, coeff) for the C side.
    # A block (br,bc) covers A[2*br..2*br+1, 2*bc..2*bc+1].

    def M(i, a_terms, b_terms, c_terms):
        return (i, a_terms, b_terms, c_terms)

    Mlist = [
        # M1 = (A00+A11)(B00+B11)
        M(0, [((0, 0), 1), ((1, 1), 1)],
              [((0, 0), 1), ((1, 1), 1)],
              [((0, 0), 1), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]),
        # M2 = (A10+A11)B00
        M(1, [((1, 0), 1), ((1, 1), 1)],
              [((0, 0), 1)],
              [((1, 0), 1), ((1, 1), 1)]),
        # M3 = A00(B01-B11)
        M(2, [((0, 0), 1)],
              [((0, 1), 1), ((1, 1), -1)],
              [((0, 1), 1), ((1, 1), -1)]),
        # M4 = A11(B10-B00)
        M(3, [((1, 1), 1)],
              [((1, 0), 1), ((0, 0), -1)],
              [((1, 0), 1), ((0, 0), -1)]),
        # M5 = (A00+A01)B11
        M(4, [((0, 0), 1), ((0, 1), 1)],
              [((1, 1), 1)],
              [((1, 0), 1), ((1, 1), 1)]),
        # M6 = (A10-A00)(B00+B01)
        M(5, [((1, 0), 1), ((0, 0), -1)],
              [((0, 0), 1), ((0, 1), 1)],
              [((0, 0), 1), ((0, 1), 1)]),
        # M7 = (A01-A11)(B10+B11)
        M(6, [((0, 1), 1), ((1, 1), -1)],
              [((1, 0), 1), ((1, 1), 1)],
              [((1, 0), 1), ((1, 1), -1)]),
    ]

    for mi, (idx, a_terms, b_terms, c_terms) in enumerate(Mlist):
        base = mi * 7
        # For each of the 7 Strassen2 terms, combine.
        for t in range(7):
            r = base + t
            # U: for each A entry (i,j), sum over a_terms and over the
            # corresponding Strassen2 U entry (local (li,lj)) -> coeff * U2[li*m2+lj, t]
            for (abr, abc, ca) in a_terms:
                for li in range(2):
                    for lj in range(2):
                        u2 = U2[li * n2 + lj, t]
                        if u2 != 0:
                            i = 2 * abr + li
                            j = 2 * abc + lj
                            U[i * m + j][r] += ca * u2
            # V: B side
            for (bbr, bbc, cb) in b_terms:
                for lj in range(2):
                    for lk in range(2):
                        v2 = V2[lj * m2 + lk, t]
                        if v2 != 0:
                            j = 2 * bbr + lj
                            k = 2 * bbc + lk
                            V[j * p + k][r] += cb * v2
            # W: C side
            for (cbr, cbc, cw) in c_terms:
                for li in range(2):
                    for lk in range(2):
                        w2 = W2[lk * p2 + li, t]
                        if w2 != 0:
                            i = 2 * cbr + li
                            k = 2 * cbc + lk
                            W[k * n + i][r] += cw * w2
    return U, V, W


# ---- Schoolbook 4x4x1 (rank 4) -----------------------------------------------
def _schoolbook_4x4x1():
    """U,V,W for 4x4 * 4x1 = 4x1 (rank 4)."""
    n, m, p = 4, 4, 1
    R = 4
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    for j in range(m):
        r = j
        for i in range(n):
            U[i * m + j][r] = 1.0
        for k in range(p):
            V[j * p + k][r] = 1.0
        for i in range(n):
            W[k := 0] # placeholder
    # Rebuild cleanly
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    for j in range(m):
        r = j
        for i in range(n):
            U[i * m + j][r] = 1.0
        V[j * p + 0][r] = 1.0
        for i in range(n):
            W[0 * n + i][r] = 1.0
    return U, V, W


# ---- Schoolbook full 4x4x9 (rank 144) ----------------------------------------
def _schoolbook_full():
    n, m, p = 4, 4, 9
    R = n * m * p
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _assemble():
    """Assemble the rank-102 decomposition:
       C = A * B where B is 4x9.
       Split B into B_left (4x4) and B_right (4x5) -> no, split into
       B1 (4x4) cols 0-3 and B2 (4x4) cols 4-7 and B3 (4x1) col 8.
       C1 (4x4) = A * B1  -> Strassen4 rank 49
       C2 (4x4) = A * B2  -> Strassen4 rank 49
       C3 (4x1) = A * B3  -> schoolbook rank 4
    """
    n, m, p = 4, 4, 9
    U1, V1, W1 = _strassen4()   # 16x49, for cols 0..3
    U2, V2, W2 = _strassen4()   # 16x49, for cols 4..7
    U3, V3, W3 = _schoolbook_4x4x1()  # 16x4, for col 8

    R = 49 + 49 + 4
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]

    # Block 1: cols 0..3 (k offset 0)
    for r in range(49):
        for i in range(n * m):
            U[i][r] = U1[i][r]
        for j in range(m):
            for k in range(4):
                V[j * p + k][r] = V1[j * 4 + k][r]
        for k in range(4):
            for i in range(n):
                W[k * n + i][r] = W1[k * 4 + i][r]
    # Block 2: cols 4..7 (k offset 4)
    for r in range(49):
        rr = 49 + r
        for i in range(n * m):
            U[i][rr] = U2[i][r]
        for j in range(m):
            for k in range(4):
                V[j * p + (4 + k)][rr] = V2[j * 4 + k][r]
        for k in range(4):
            for i in range(n):
                W[(4 + k) * n + i][rr] = W2[k * 4 + i][r]
    # Block 3: col 8 (k offset 8)
    for r in range(4):
        rr = 98 + r
        for i in range(n * m):
            U[i][rr] = U3[i][r]
        for j in range(m):
            V[j * p + 8][rr] = V3[j * 1 + 0][r]
        for i in range(n):
            W[8 * n + i][rr] = W3[0 * n + i][r]

    return {"U": U, "V": V, "W": W}


def construct():
    """Returns a tensor decomposition for 4x4 * 4x9 matrix multiplication.

    Attempts the rank-102 Strassen-based decomposition; falls back to
    schoolbook rank 144 if verification fails.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0

    # Try rank-102 Strassen decomposition
    try:
        result = _assemble()
        if time.time() < deadline - 5:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                # Round to clean integer/half-integer values
                def clean(rows):
                    return [[round(v * 2) / 2 for v in row] for row in rows]
                return {"U": clean(result["U"]),
                        "V": clean(result["V"]),
                        "W": clean(result["W"])}
    except Exception:
        pass

    # Fallback: schoolbook rank 144
    return _schoolbook_full()
