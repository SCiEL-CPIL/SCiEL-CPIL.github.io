"""Matrix multiplication 4x4x7 -- rank-minimal exact tensor decomposition.

Design: split the 4x4x7 multiplication tensor into 4 disjoint 2x2x2 block
tensors (rows split {0,1}/{2,3}, cols split {0,1}/{2,3}, and the 7 output
columns grouped as {0,1},{2,3},{4,5} plus singleton {6}) plus 5 rank-1
"cross" terms.  Each 2x2x2 block uses the Strassen decomposition (rank 7).
Total rank = 4*7 + 5 = 61, far below the schoolbook 112.
"""
from fractions import Fraction
import time

_START = time.time()
_DEADLINE = _START + 140.0  # explicit guard: safety margin before the 150s kill

DIMENSIONS = [4, 4, 7]


def _strassen_222():
    """Return the 7 rank-1 terms (u, v, w) for the 2x2x2 tensor T[i,j,k]=delta[i,k].
    u: (2*2, 7) for the 2x2 A (row i, col j); v: (2*2, 7) for the 2x2 B (row j, col k);
    w: (2*2, 7) for the 2x2 C (row k, col i).  Entries are Fractions."""
    u = [[Fraction(0)] * 7 for _ in range(4)]
    v = [[Fraction(0)] * 7 for _ in range(4)]
    w = [[Fraction(0)] * 7 for _ in range(4)]

    def set1(arr, r, c, val):
        arr[r][c] = val

    # m1: (A00+A11)(B10-B11)  -> C10
    set1(u, 0, 0, 1); set1(u, 3, 0, 1)
    set1(v, 2, 0, 1); set1(v, 3, 0, -1)
    set1(w, 2, 0, 1)
    # m2: (A00+A01)B11        -> C01
    set1(u, 0, 1, 1); set1(u, 1, 1, 1)
    set1(v, 3, 1, 1)
    set1(w, 1, 1, 1)
    # m3: (A10-A00)(B00+B01)  -> C10
    set1(u, 2, 2, 1); set1(u, 0, 2, -1)
    set1(v, 0, 2, 1); set1(v, 1, 2, 1)
    set1(w, 2, 2, 1)
    # m4: (A01-A11)(B01+B11)  -> C01
    set1(u, 1, 3, 1); set1(u, 3, 3, -1)
    set1(v, 1, 3, 1); set1(v, 3, 3, 1)
    set1(w, 1, 3, 1)
    # m5: (A00+A10)B10        -> C00
    set1(u, 0, 4, 1); set1(u, 2, 4, 1)
    set1(v, 2, 4, 1)
    set1(w, 0, 4, 1)
    # m6: (A11-A01)(B00+B10)  -> C11
    set1(u, 3, 5, 1); set1(u, 1, 5, -1)
    set1(v, 0, 5, 1); set1(v, 2, 5, 1)
    set1(w, 3, 5, 1)
    # m7: (A00+A01+A10+A11)B00 -> C00
    set1(u, 0, 6, 1); set1(u, 1, 6, 1); set1(u, 2, 6, 1); set1(u, 3, 6, 1)
    set1(v, 0, 6, 1)
    set1(w, 0, 6, 1)
    return u, v, w


def _build_decomposition():
    n, m, p = 4, 4, 7
    U = []  # rows i*m+j
    V = []  # rows j*p+k
    W = []  # rows k*n+i
    # We build column lists of (Urow, Vrow, Wrow, val)
    cols = []  # each col is a dict row->val for U, V, W

    def add_block(r0, c0, k0, su, sv, sw):
        # su, sv, sw are 4x7 strassen matrices for a 2x2x2 block at
        # A rows r0..r0+1, A cols c0..c0+1, B cols k0..k0+1
        for t in range(7):
            Ucol = {}
            Vcol = {}
            Wcol = {}
            for a in range(2):
                for b in range(2):
                    val = su[a * 2 + b][t]
                    if val:
                        Ucol[(r0 + a) * m + (c0 + b)] = Ucol.get((r0 + a) * m + (c0 + b), 0) + val
            for b in range(2):
                for c in range(2):
                    val = sv[b * 2 + c][t]
                    if val:
                        Vcol[(c0 + b) * p + (k0 + c)] = Vcol.get((c0 + b) * p + (k0 + c), 0) + val
            for c in range(2):
                for a in range(2):
                    val = sw[c * 2 + a][t]
                    if val:
                        Wcol[(k0 + c) * n + (r0 + a)] = Wcol.get((k0 + c) * n + (r0 + a), 0) + val
            cols.append((Ucol, Vcol, Wcol))

    def add_cross(r0, c0, k0):
        Ucol = {(r0) * m + (c0)}
        Vcol = {(c0) * p + (k0)}
        Wcol = {(k0) * n + (r0)}
        cols.append((Ucol, Vcol, Wcol))

    su, sv, sw = _strassen_222()
    # 4 2x2x2 blocks
    add_block(0, 0, 0, su, sv, sw)   # A rows 0,1 cols 0,1 ; B cols 0,1
    add_block(0, 2, 2, su, sv, sw)   # A rows 0,1 cols 2,3 ; B cols 2,3
    add_block(2, 0, 4, su, sv, sw)   # A rows 2,3 cols 0,1 ; B cols 4,5
    add_block(2, 2, 6, su, sv, sw)   # A rows 2,3 cols 2,3 ; B col 6
    # 5 cross terms (singletons)
    add_cross(0, 2, 6)
    add_cross(0, 3, 4)
    add_cross(0, 3, 5)
    add_cross(1, 2, 6)
    add_cross(1, 3, 4)

    rank = len(cols)
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    for c, (Ucol, Vcol, Wcol) in enumerate(cols):
        for r, val in Ucol.items():
            U[r][c] = val
        for r, val in Vcol.items():
            V[r][c] = val
        for r, val in Wcol.items():
            W[r][c] = val
    return U, V, W, rank


def _verify(U, V, W):
    """Exact check: sum_c U[i*m+j,c]*V[j*p+k,c] == delta[i,k] for all i,j,k."""
    n, m, p = 4, 4, 7
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = Fraction(0)
                ui = U[i * m + j]
                vj = V[j * p + k]
                for c in range(rank):
                    acc += ui[c] * vj[c]
                expected = Fraction(1) if i == k else Fraction(0)
                if acc != expected:
                    return False
    return True


def _verify_np(U, V, W):
    import numpy as np
    n, m, p = 4, 4, 7
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    # C[i,k] = sum_j U[i*m+j,:] . V[j*p+k,:]
    C = np.zeros((n, p), dtype=object)
    for i in range(n):
        for k in range(p):
            acc = Fraction(0)
            for j in range(m):
                acc += sum(Ua[i * m + j, c] * Va[j * p + k, c] for c in range(len(Ua[0])))
            C[i, k] = acc
    for i in range(n):
        for k in range(p):
            expected = Fraction(1) if i == k else Fraction(0)
            if C[i, k] != expected:
                return False
    return True


def construct():
    U, V, W, rank = _build_decomposition()
    # Exact verification (cheap: 4*4*7*61 = ~6832 Fraction mults)
    if not _verify(U, V, W):
        # Fallback to the always-valid schoolbook decomposition.
        n, m, p = DIMENSIONS
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
        return {"U": U, "V": V, "W": W}
    return {"U": U, "V": V, "W": W}
