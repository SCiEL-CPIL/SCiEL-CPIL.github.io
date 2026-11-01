"""Tensor decomposition search for 2x2 by 2x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS


def _verify_bilinear(U, V, W, n, m, p, tol=1e-10):
    """Verify by checking the bilinear form reproduces matrix multiplication."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for r in range(len(W[0])):
                lA = sum(U[i * m + j][r] * A[i, j] for j in range(m))
                lB = sum(V[j * p + k][r] * B[j, k] for j in range(m) if j < m)
                # V is indexed by j*p+k where j ranges over the shared dimension
                # For B[j,k], j is the row of B (0 to m-1), k is the column (0 to p-1)
                C_computed[i, k] += W[k * n + i][r] * lA * lB
    return np.allclose(C_computed, C_expected, atol=tol)


def _verify_tensor(U, V, W, n, m, p, tol=1e-10):
    """Verify that the tensor decomposition matches: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_exact_integer(U, V, W, n, m, p):
    """Verify using exact integer arithmetic (all entries must be integers or half-integers)."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    u8 = int(round(U[i * m + j][r] * 8))
                    v8 = int(round(V[j * p + k][r] * 8))
                    w8 = int(round(W[k * n + i][r] * 8))
                    s += u8 * v8 * w8
                if s != 512:
                    return False
    return True


def _rank11_decomposition():
    """
    Rank-11 decomposition for 2x2 x 2x3 matrix multiplication.
    7 Strassen-like terms for the 2x2x2 core + 4 schoolbook terms for column 2.
    All entries are integers.
    """
    rank = 11
    U = [[0] * rank for _ in range(4)]  # n*m = 4
    V = [[0] * rank for _ in range(6)]  # m*p = 6
    W = [[0] * rank for _ in range(6)]  # p*n = 6

    # Strassen part for columns 0 and 1 of B
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1

    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1

    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1

    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1

    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1

    # W for Strassen outputs
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C01 = M3 + M5
    W[2][2] = 1; W[2][4] = 1
    # C10 = M2 + M4
    W[1][1] = 1; W[1][3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1

    # Schoolbook part for column 2 of B
    # M8 = A00 * B02
    U[0][7] = 1
    V[2][7] = 1

    # M9 = A01 * B12
    U[1][8] = 1
    V[5][8] = 1

    # M10 = A10 * B02
    U[2][9] = 1
    V[2][9] = 1

    # M11 = A11 * B12
    U[3][10] = 1
    V[5][10] = 1

    # W for Schoolbook outputs
    # C02 = M8 + M9
    W[4][7] = 1; W[4][8] = 1
    # C12 = M10 + M11
    W[5][9] = 1; W[5][10] = 1

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 12."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Return the rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication.
    The known minimal rank is 11 (Strassen's 7 + 4 schoolbook for the third column).
    """
    # Return the verified rank-11 decomposition immediately
    result11 = _rank11_decomposition()

    # Verify with exact integer arithmetic first (fastest, no FP issues)
    if _verify_exact_integer(result11["U"], result11["V"], result11["W"], n, m, p):
        return result11

    # Fallback: verify with tensor equation
    if _verify_tensor(result11["U"], result11["V"], result11["W"], n, m, p):
        return result11

    # Fallback: verify with bilinear form
    if _verify_bilinear(result11["U"], result11["V"], result11["W"], n, m, p):
        return result11

    # Last resort: schoolbook (rank 12)
    return _schoolbook()
