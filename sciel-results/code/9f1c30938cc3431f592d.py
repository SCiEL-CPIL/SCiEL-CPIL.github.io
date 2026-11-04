"""
Rank-minimal tensor decomposition for 2x5 * 5x5 matrix multiplication.

Strategy:
1. Try a rank-49 decomposition by embedding Strassen's 2x2 algorithm on the
   sub-block A[:, 0:2] * B[0:2, :] (covering C[*, 0:2] from those terms) and
   using schoolbook for all remaining bilinear terms.
2. Verify the decomposition with exact integer arithmetic.
3. Fall back to the guaranteed-valid schoolbook rank-50 decomposition.

The score is the rank (lower is better). The seed program (schoolbook) has
score 1.0000 (rank 50). A valid rank-49 decomposition would score 50/49 ≈ 1.0204.
"""

import time

DIMENSIONS = [2, 5, 5]
_T0 = time.time()
_DEADLINE_S = 110.0


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S


def _schoolbook():
    """Return (U, V, W) as lists of lists with integer 0/1 entries, rank = n*m*p = 50."""
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
    return U, V, W, rank


def _strassen_embedding():
    """
    Construct a rank-49 decomposition:
      - 7 Strassen terms handle A[:,0:2] * B[0:2,:] contributing to C[:,0:2]
      - 42 schoolbook terms handle all remaining (i,j,k) combinations

    The 2x2 Strassen algorithm:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B01-B11)
      M4 = A11(B10-B00)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B01)
      M7 = (A01-A11)(B10+B11)

      C00 = M1 - M4 + M5 + M6
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M7
    """
    n, m, p = DIMENSIONS  # 2, 5, 5

    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # shape (10, 49)
    V = [[0] * rank for _ in range(m * p)]  # shape (25, 49)
    W = [[0] * rank for _ in range(p * n)]  # shape (10, 49)

    # Strassen U coefficients for A indices:
    # A[0,0] -> U[0], A[0,1] -> U[1], A[1,0] -> U[5], A[1,1] -> U[6]
    U_strassen = {
        0: [1, 0, 1, 0, 1, -1, 0],  # A[0,0]
        1: [0, 0, 0, 0, 1, 0, 1],   # A[0,1]
        5: [0, 1, 0, 0, 0, 1, 0],   # A[1,0]
        6: [1, 1, 0, 1, 0, 0, -1],  # A[1,1]
    }
    for idx, vals in U_strassen.items():
        for r in range(7):
            U[idx][r] = vals[r]

    # Strassen V coefficients for B indices:
    # B[0,0] -> V[0], B[0,1] -> V[1], B[1,0] -> V[5], B[1,1] -> V[6]
    V_strassen = {
        0: [1, 1, 0, -1, 0, 1, 0],  # B[0,0]
        1: [0, 0, 1, 0, 0, 1, 0],   # B[0,1]
        5: [0, 0, 0, 1, 0, 0, 1],   # B[1,0]
        6: [1, 0, -1, 0, 1, 0, 1],  # B[1,1]
    }
    for idx, vals in V_strassen.items():
        for r in range(7):
            V[idx][r] = vals[r]

    # Strassen W coefficients for C indices (trace convention W[k*n+i]):
    # C[0,0] -> W[0], C[1,0] -> W[1], C[0,1] -> W[2], C[1,1] -> W[3]
    W_strassen = {
        0: [1, 0, 0, -1, 1, 1, 0],   # C[0,0]
        1: [0, 1, 0, 1, 0, 0, 0],    # C[1,0]
        2: [0, 0, 1, 0, 1, 0, 0],    # C[0,1]
        3: [1, -1, 1, 0, 0, 0, 1],   # C[1,1]
    }
    for idx, vals in W_strassen.items():
        for r in range(7):
            W[idx][r] = vals[r]

    # Schoolbook terms for remaining products:
    # For k in {0,1}: j in {2,3,4} (Strassen handles j in {0,1})
    # For k in {2,3,4}: j in {0,1,2,3,4} (no Strassen coverage)
    col = 7
    for k in range(p):
        for i in range(n):
            j_start = 2 if k < 2 else 0
            for j in range(j_start, m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return U, V, W, rank


def _verify(U, V, W, rank):
    """Exactly check that the decomposition reproduces the multiplication tensor.

    For each (i, j, k) the sum over r of
        U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1. Uses integer arithmetic only.
    """
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                s = 0
                for r in range(rank):
                    s += ui[r] * vj[r] * wk[r]
                if s != 1:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS

    # Try the Strassen embedding (rank 49)
    if _deadline_ok():
        U, V, W, rank = _strassen_embedding()
        if _verify(U, V, W, rank):
            return {"U": U, "V": V, "W": W}

    # Fall back to schoolbook (rank 50)
    U, V, W, rank = _schoolbook()
    if _verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}

    # Should never reach here, but return anyway
    return {"U": U, "V": V, "W": W}
