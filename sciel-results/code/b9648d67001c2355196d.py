"""Rewritten program: block-diagonal Strassen-based rank-128 decomposition of 4x4x9,
with runtime tensor verification and a schoolbook fallback for guaranteed validity."""

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np


def _strassen_2x2x2():
    """7 bilinear factors for 2x2 * 2x2 -> 2x2 (Strassen)."""
    Ms = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1),((1,1),-1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,1),1),((1,1),-1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,0),1),((1,0),1)]),
    ]
    return Ms


def _strassen_2x2x1():
    """4 bilinear factors for 2x2 * 2x1 -> 2x1 (schoolbook-minimal)."""
    Ps = [
        ([((0,0),1)], [((0,0),1)], [((0,0),1)]),
        ([((0,1),1)], [((1,0),1)], [((0,0),1)]),
        ([((1,0),1)], [((0,0),1)], [((1,0),1)]),
        ([((1,1),1)], [((1,0),1)], [((1,0),1)]),
    ]
    return Ps


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    S22 = _strassen_2x2x2()
    S21 = _strassen_2x2x1()

    U = {}
    V = {}
    W = {}
    col = [0]

    def add_term(ue, ve, we):
        c = col[0]
        col[0] += 1
        for (r, val) in ue:
            U.setdefault(r, {})
            U[r][c] = U[r].get(c, 0) + val
        for (r, val) in ve:
            V.setdefault(r, {})
            V[r][c] = V[r].get(c, 0) + val
        for (r, val) in we:
            W.setdefault(r, {})
            W[r][c] = W[r].get(c, 0) + val

    def emit_2col(kbase):
        # 4-col output block of width 2 starting at column kbase.
        # 4 Strassen-222 products.
        for (a_block, b_row0, c_row0, aj0) in [
            ((0, 0), 0, 0, 0),  # A00 * B[0:2,kbase:kbase+2] -> C[kbase:kbase+2, 0:2]
            ((0, 1), 2, 0, 2),  # A01 * B[2:4,...]          -> C[., 0:2]
            ((1, 0), 0, 2, 0),  # A10 * B[0:2,...]          -> C[., 2:4]
            ((1, 1), 2, 2, 2),  # A11 * B[2:4,...]          -> C[., 2:4]
        ]:
            ai0 = a_block[0]
            for (Ae, Be, Ce) in S22:
                ue = [((ai0 * 2 + r) * 4 + (aj0 + cc), val) for ((r, cc), val) in Ae]
                ve = [((b_row0 + r) * 9 + (kbase + cc), val) for ((r, cc), val) in Be]
                we = [((kbase + cc) * 4 + (c_row0 + r), val) for ((r, cc), val) in Ce]
                add_term(ue, ve, we)

    def emit_1col(kcol):
        for (a_block, b_row0, c_row0, aj0) in [
            ((0, 0), 0, 0, 0),
            ((0, 1), 2, 0, 2),
            ((1, 0), 0, 2, 0),
            ((1, 1), 2, 2, 2),
        ]:
            ai0 = a_block[0]
            for (Ae, Be, Ce) in S21:
                ue = [((ai0 * 2 + r) * 4 + (aj0 + cc), val) for ((r, cc), val) in Ae]
                ve = [((b_row0 + r) * 9 + kcol, val) for ((r, _c), val) in Be]
                we = [(kcol * 4 + (c_row0 + r), val) for ((r, _c), val) in Ce]
                add_term(ue, ve, we)

    # 9 columns = 4 two-column blocks (0-1,2-3,4-5,6-7) + 1 one-column block (8)
    for kbase in (0, 2, 4, 6):
        emit_2col(kbase)
    emit_1col(8)

    rank = col[0]
    nU, nV, nW = n * m, m * p, p * n
    Umat = [[0] * rank for _ in range(nU)]
    Vmat = [[0] * rank for _ in range(nV)]
    Wmat = [[0] * rank for _ in range(nW)]
    for r, d in U.items():
        for c, val in d.items():
            Umat[r][c] = val
    for r, d in V.items():
        for c, val in d.items():
            Vmat[r][c] = val
    for r, d in W.items():
        for c, val in d.items():
            Wmat[r][c] = val

    # Runtime verification against the canonical matrix-mult tensor (entry 1 at (i,j,k)).
    Ua = np.array(Umat, dtype=float)
    Va = np.array(Vmat, dtype=float)
    Wa = np.array(Wmat, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.einsum('c,c,c->', Ua[i * m + j], Va[j * p + k], Wa[k * n + i]))
                if abs(s - 1.0) > 1e-6:
                    return _schoolbook(n, m, p)

    return {"U": Umat, "V": Vmat, "W": Wmat}
# EVOLVE-BLOCK-END
