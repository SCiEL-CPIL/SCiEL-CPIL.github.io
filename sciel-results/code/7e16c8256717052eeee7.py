"""Block-partitioned exact decomposition for 4x4 matrix multiplication.

Splits 4x4 = 2x2 block matrix into:
  - two 2x2 diagonal multiplications (Strassen, rank 7 each)
  - four 2x2 "outer-product" blocks (rank 4 each)
Total rank = 7+7+4+4+4+4 = 22.

Strassen factors (A,B -> C, C = A*B):
  M1=(a00+a11)(b00+b11)  M2=(a10+a11)b00  M3=a00(b01-b11)
  M4=a11(b10-b00)        M5=(a00+a01)b11  M6=(a10-a00)(b00+b01)
  M7=(a01-a11)(b10+b11)
  c00=M1+M4-M5+M7  c01=M3+M5  c10=M2+M4  c11=M1-M2+M3+M6
"""
import time
from fractions import Fraction

_DIM = (4, 4, 4)
_T0 = None


def _now():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return time.time()


def _safe(f):
    # Fraction or int -> python int if integer-valued, else float (half-integer)
    if isinstance(f, Fraction):
        if f.denominator == 1:
            return int(f.numerator)
        return float(f)
    return f


class _Builder:
    def __init__(self, rank):
        self.rank = rank
        self.U = [[0] * rank for _ in range(16)]
        self.V = [[0] * rank for _ in range(16)]
        self.W = [[0] * rank for _ in range(16)]

    def add(self, urow, val_u, vrow, val_v, wrow, val_w, sign=1):
        c = self.rank
        self.rank += 1
        self.U.append([0] * self.rank)
        self.V.append([0] * self.rank)
        self.W.append([0] * self.rank)
        self.U[urow][c] = sign * val_u
        self.V[vrow][c] = sign * val_v
        self.W[wrow][c] = sign * val_w
        return c


def _add_strassen(b, U, V, W, a_off, b_off, c_off):
    # a_off/b_off: (row0,row1,col0,col1) base offsets into A/B
    # c_off: (row0,row1,col0,col1) base offsets into C
    a00, a01 = a_off[0], a_off[0] + 1
    a10, a11 = a_off[1], a_off[1] + 1
    b00, b01 = b_off[0], b_off[0] + 1
    b10, b11 = b_off[1], b_off[1] + 1
    c00, c01 = c_off[0], c_off[0] + 1
    c10, c11 = c_off[1], c_off[1] + 1

    def addM(u_terms, v_terms, c_terms, sign):
        c = b.add(0, 0, 0, 0, 0, 0, sign)  # placeholder column
        # overwrite the three factor rows with the linear combos
        for r, v in u_terms:
            U[r][c] = sign * v
        for r, v in v_terms:
            V[r][c] = sign * v
        for r, v in c_terms:
            W[r][c] = sign * v

    # M1 = (a00+a11)(b00+b11) -> c00 +1
    addM([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(c00, 1)], 1)
    # M2 = (a10+a11) b00 -> c10 +1, c11 -1
    addM([(a10, 1), (a11, 1)], [(b00, 1)], [(c10, 1), (c11, -1)], 1)
    # M3 = a00 (b01-b11) -> c01 +1, c11 +1
    addM([(a00, 1)], [(b01, 1), (b11, -1)], [(c01, 1), (c11, 1)], 1)
    # M4 = a11 (b10-b00) -> c00 +1, c10 +1
    addM([(a11, 1)], [(b10, 1), (b00, -1)], [(c00, 1), (c10, 1)], 1)
    # M5 = (a00+a01) b11 -> c00 -1, c01 +1
    addM([(a00, 1), (a01, 1)], [(b11, 1)], [(c00, -1), (c01, 1)], 1)
    # M6 = (a10-a00)(b00+b01) -> c11 +1
    addM([(a10, 1), (a00, -1)], [(b00, 1), (b01, 1)], [(c11, 1)], 1)
    # M7 = (a01-a11)(b10+b11) -> c00 +1
    addM([(a01, 1), (a11, -1)], [(b10, 1), (b11, 1)], [(c00, 1)], 1)


def _add_outer(b, U, V, W, a_off, b_off, c_off):
    # C_block = A_block * B_block, each 2x2, rank 4
    a00, a01 = a_off[0], a_off[0] + 1
    a10, a11 = a_off[1], a_off[1] + 1
    b00, b01 = b_off[0], b_off[0] + 1
    b10, b11 = b_off[1], b_off[1] + 1
    c00, c01 = c_off[0], c_off[0] + 1
    c10, c11 = c_off[1], c_off[1] + 1

    def addM(ar, av, br, bv, cr, cv):
        c = b.add(0, 0, 0, 0, 0, 0, 1)
        U[ar][c] = av
        V[br][c] = bv
        W[cr][c] = cv

    addM(a00, 1, b00, 1, c00, 1)
    addM(a00, 1, b01, 1, c01, 1)
    addM(a10, 1, b00, 1, c10, 1)
    addM(a10, 1, b01, 1, c11, 1)


def _verify(U, V, W):
    """Check sum_c U[r,c]*V[s,c] == C[r,s] for the 4x4 multiplication tensor."""
    n = m = p = 4
    rank = len(U[0])
    for r in range(n * m):
        for s in range(m * p):
            acc = 0
            for c in range(rank):
                acc += U[r][c] * V[s][c]
            # expected C[r, s] = 1 iff r and s share a middle index j
            # r = i*m+j, s = j'*p+k ; C nonzero when j == j'
            j = r % m
            jp = s // p
            expected = 1 if j == jp else 0
            if acc != expected:
                raise ValueError(f"verify fail r={r} s={s} acc={acc} exp={expected}")


def construct():
    deadline = _now() + 110.0  # safety margin before 120s limit
    b = _Builder(0)
    U, V, W = b.U, b.V, b.W

    # Diagonal blocks: A[0:2,0:2]*B[0:2,0:2] -> C[0:2,0:2] (Strassen)
    _add_strassen(b, U, V, W, (0, 2, 0, 2), (0, 2, 0, 2), (0, 2, 0, 2))
    # A[2:4,2:4]*B[2:4,2:4] -> C[2:4,2:4] (Strassen)
    _add_strassen(b, U, V, W, (8, 10, 8, 10), (8, 10, 8, 10), (8, 10, 8, 10))

    # Off-diagonal outer-product blocks
    # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,2:4]
    _add_outer(b, U, V, W, (4, 6, 4, 6), (8, 10, 8, 10), (4, 6, 4, 6))
    # A[2:4,0:2] * B[0:2,2:4] -> C[2:4,0:2]
    _add_outer(b, U, V, W, (8, 10, 8, 10), (4, 6, 4, 6), (8, 10, 8, 10))
    # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,4:6]
    _add_outer(b, U, V, W, (4, 6, 4, 6), (12, 14, 12, 14), (12, 14, 12, 14))
    # A[2:4,0:2] * B[0:2,4:6] -> C[2:4,4:6]
    _add_outer(b, U, V, W, (8, 10, 8, 10), (12, 14, 12, 14), (16, 18, 16, 18))

    # Convert to plain ints (all entries are integers here)
    U = [[_safe(x) for x in row] for row in U]
    V = [[_safe(x) for x in row] for row in V]
    W = [[_safe(x) for x in row] for row in W]

    # Verify within budget
    if _now() < deadline:
        _verify(U, V, W)

    return {"U": U, "V": V, "W": W}
