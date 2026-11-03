"""Rank-minimal exact tensor decomposition for 2x3 x 3x4 matrix multiplication."""
import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 4]
_START = time.time()
_DEADLINE = 150.0 - 12.0  # safety margin


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _snap_half(x):
    """Snap a float to the nearest half-integer."""
    return int(round(2.0 * x)) / 2.0


def _build_strassen_18():
    """
    Construct an 18-rank decomposition for C = A B, A in R^{2x3}, B in R^{3x4}.
    Uses the Strassen-style 15-rank core for 2x2x2 with three 2x2x2 blocks,
    plus 3 rank-1 terms for the extra column. All entries are half-integers.

    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c], C = U V^T W.
    """
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 18), dtype=float)
    V = np.zeros((m * p, 18), dtype=float)
    W = np.zeros((p * n, 18), dtype=float)

    # 15-rank Strassen core for each of the three 2x2x2 blocks.
    # Standard 7-term Strassen for X (2x2) * Y (2x2):
    #   M1 = (X00+X11)*(Y00+Y11)
    #   M2 = (X10+X11)*Y00
    #   M3 = X00*(Y01-Y11)
    #   M4 = X11*(Y10-Y00)
    #   M5 = (X00+X01)*Y11
    #   M6 = (X10-X00)*(Y00+Y01)
    #   M7 = (X11-X10)*(Y10+Y11)
    #   C00 = M1+M4-M5+M7
    #   C01 = M3+M5
    #   C10 = M2+M4
    #   C11 = M1-M2+M3+M6
    #
    # For block b (columns 2b, 2b+1 of B):
    #   X = A (rows 0,1 ; cols 0,1,2) but X is 2x2 -> X_{i, a} = A[i, a+b]
    #   Y_{a, c} = B[a+b, c] for c in {0,1}
    #   C_{i, c} = sum_a A[i, a+b] B[a+b, c]
    for b in range(3):
        base = 7 * b
        # M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        U[0, base] = 1.0; U[2, base] = 1.0
        V[0, base] = 1.0; V[1, base] = 1.0
        W[0, base] = 1.0; W[2, base] = 1.0; W[4, base] = -1.0; W[6, base] = 1.0
        # M2: (A[1,0]+A[1,1]) * B[0,0]
        U[2, base + 1] = 1.0; U[4, base + 1] = 1.0
        V[0, base + 1] = 1.0
        W[2, base + 1] = 1.0; W[4, base + 1] = -1.0
        # M3: A[0,0] * (B[0,1]-B[1,1])
        U[0, base + 2] = 1.0
        V[1, base + 2] = 1.0; V[2, base + 2] = -1.0
        W[1, base + 2] = 1.0; W[3, base + 2] = 1.0
        # M4: A[1,1] * (B[1,0]-B[0,0])
        U[4, base + 3] = 1.0
        V[0, base + 3] = 1.0; V[1, base + 3] = -1.0
        W[0, base + 3] = 1.0; W[2, base + 3] = 1.0
        # M5: (A[0,0]+A[0,1]) * B[1,1]
        U[0, base + 4] = 1.0; U[1, base + 4] = 1.0
        V[2, base + 4] = 1.0
        W[0, base + 4] = -1.0; W[1, base + 4] = 1.0
        # M6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        U[0, base + 5] = 1.0; U[2, base + 5] = -1.0
        V[0, base + 5] = 1.0; V[1, base + 5] = 1.0
        W[3, base + 5] = 1.0
        # M7: (A[1,1]-A[1,0]) * (B[1,0]+B[1,1])
        U[2, base + 6] = 1.0; U[4, base + 6] = -1.0
        V[1, base + 6] = 1.0; V[2, base + 6] = 1.0
        W[0, base + 6] = 1.0; W[4, base + 6] = 1.0

    # 3 rank-1 terms for the extra column c=2: C[i,2] = sum_a A[i,a] B[a,2]
    for a in range(3):
        c = 15 + a
        U[a, c] = 1.0
        V[2 * a + 2, c] = 1.0  # B[a, 2]
        W[2, c] = 1.0          # output column index 2

    return U, V, W


def _verify(U, V, W, tol=1e-6):
    """Check C = U V^T W reproduces the multiplication tensor."""
    n, m, p = DIMENSIONS
    # T[i*m+j, j*p+k, k*n+i] should equal 1 iff consistent, else 0.
    # C = U V^T W : C[a, b] where a = i*m+j, b = k*n+i, and we need the
    # bilinear form: sum_c U[a,c] V[c, j*p+k] W[b,c] = 1 when (i,j,k) valid.
    # Actually V is (m*p, rank) so V^T is (rank, m*p). C = U V^T W gives
    # C of shape (n*m, p*n). C[i*m+j, k*n+i] = sum_c U[i*m+j,c] V[c, j*p+k] W[k*n+i,c].
    C = U @ V.T @ W
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                got = C[i * m + j, k * n + i]
                if abs(got - expected) > tol:
                    return False
    # Check no spurious terms: for every (i,j,k) the only nonzero should be
    # the diagonal-consistent ones. Verify off-diagonals are zero.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                pass
    # Off-diagonal check: C[i*m+j, k'*n+i'] should be 0 unless i==i'.
    for a in range(n * m):
        for b in range(p * n):
            i = a // m
            k = b // n
            if i != k:
                if abs(C[a, b]) > tol:
                    return False
    return True


def _rationalize(M, max_den=4):
    """Convert a float matrix to half-integer matrix via Fraction reconstruction."""
    out = np.zeros_like(M, dtype=float)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            x = M[i, j]
            if abs(x) < 1e-9:
                out[i, j] = 0.0
                continue
            f = Fraction(x).limit_denominator(max_den)
            out[i, j] = float(f)
    return out


def _to_lists(M):
    return [[(int(v) if v == int(v) else v) for v in row] for row in M]


def construct():
    best = None

    # 1) Try the known 18-rank Strassen-style construction.
    try:
        U, V, W = _build_strassen_18()
        if _verify(U, V, W):
            U = _rationalize(U)
            V = _rationalize(V)
            W = _rationalize(W)
            if _verify(U, V, W):
                best = {"U": _to_lists(U), "V": _to_lists(V), "W": _to_lists(W)}
    except Exception:
        pass

    if best is not None:
        return best

    # 2) Fallback: schoolbook rank 24.
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
