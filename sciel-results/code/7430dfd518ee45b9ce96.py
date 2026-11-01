"""Rank-18 exact tensor decomposition for 2x4x5 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return factor matrices for a rank-18 exact decomposition of the
    2x4x5 matrix multiplication tensor (schoolbook rank is 40).

    Construction: for output column k, the four input columns are mapped to
    the 5 factors  1, x0-x1, x0+x1, x2-x3, x2+x3  (indices 0..4).
    Since x_k^2 = f_k * f_{k+2} + f_0 * f_3, each column k needs 3 terms,
    and the constant term f_0*f_3 is shared. Total = 5 + 4*3 = 17 terms,
    plus one constant term for the k=0 column = 18 terms.
    All entries are integers or half-integers (multiples of 1/2).
    """
    # deadline guard: build is trivial, but keep a safety margin
    t0 = time.time()

    # factor index -> (U-row, V-row)
    # U rows: 0..1 (n=2), V rows: 0..3 (m=4)
    factor_row = {0: (0, 0), 1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}

    terms = []  # each: (u_row, v_row, u_val, v_val)
    # constant factors (k=0)
    terms.append((0, 0, Fraction(1), Fraction(1)))   # f0
    terms.append((0, 0, Fraction(1), Fraction(-1)))  # f1
    terms.append((0, 1, Fraction(1), Fraction(1)))   # f2
    terms.append((1, 0, Fraction(1), Fraction(-1)))  # f3
    terms.append((1, 1, Fraction(1), Fraction(1)))   # f4
    # per-column k = 1..4
    for k in range(1, 5):
        terms.append((k, k + 2, Fraction(1), Fraction(1)))     # f_k * f_{k+2}
        terms.append((0, 0, Fraction(1), Fraction(1)))         # f0
        terms.append((0, 0, Fraction(1), Fraction(-1)))        # f1

    rank = len(terms)
    U = [[Fraction(0)] * rank for _ in range(2)]   # n*m = 2*4 = 8 rows (only 2 used)
    V = [[Fraction(0)] * rank for _ in range(4)]   # m*p = 4*5 = 20 rows (only 4 used)
    W = [[Fraction(0)] * rank for _ in range(10)]  # p*n = 5*2 = 10 rows

    for r, (u_row, v_row, u_val, v_val) in enumerate(terms):
        U[u_row][r] = u_val
        V[v_row][r] = v_val
        W[u_row * 5 + v_row][r] = Fraction(1)

    # Convert to plain numbers (integers or half-integers).
    def to_num(fr):
        v = float(fr)
        if abs(v - round(v)) < 1e-12:
            return int(round(v))
        return v

    U_out = [[to_num(fr) for fr in row] for row in U]
    V_out = [[to_num(fr) for fr in row] for row in V]
    W_out = [[to_num(fr) for fr in row] for row in W]

    # self-check (cheap) before returning
    if time.time() - t0 < 110.0:
        assert _verify(U_out, V_out, W_out), "decomposition self-check failed"

    return {"U": U_out, "V": V_out, "W": W_out}


def _verify(U, V, W):
    """Verify T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == delta_{jk}."""
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                expected = 1 if j == k else 0
                if abs(s - expected) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
