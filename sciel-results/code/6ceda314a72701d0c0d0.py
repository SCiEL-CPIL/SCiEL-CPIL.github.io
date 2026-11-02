"""Rewritten program for 2x2 matrix multiplication rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the verified rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Rank 7 is the proven minimum tensor rank for 2x2 matmul, so no lower rank is possible.
    """
    _start = time.time()
    _deadline = 120.0  # safety margin before 150s kill

    n, m, p = 2, 2, 2
    rank = 7

    # U[i*m + a, r] where i is row of A, a is col of A
    # V[a*p + k, r] where a is row of B, k is col of B
    # W[k*n + i, r] where k is col of C, i is row of C

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # M1 = (A00 + A11)(B00 + B11)
    U[0*2+0][0] = 1  # A00
    U[1*2+1][0] = 1  # A11
    V[0*2+0][0] = 1  # B00
    V[1*2+1][0] = 1  # B11

    # M2 = (A10 + A11)B00
    U[1*2+0][1] = 1  # A10
    U[1*2+1][1] = 1  # A11
    V[0*2+0][1] = 1  # B00

    # M3 = A00(B01 - B11)
    U[0*2+0][2] = 1  # A00
    V[0*2+1][2] = 1  # B01
    V[1*2+1][2] = -1 # -B11

    # M4 = A11(B10 - B00)
    U[1*2+1][3] = 1  # A11
    V[1*2+0][3] = 1  # B10
    V[0*2+0][3] = -1 # -B00

    # M5 = (A00 + A01)B11
    U[0*2+0][4] = 1  # A00
    U[0*2+1][4] = 1  # A01
    V[1*2+1][4] = 1  # B11

    # M6 = (A10 - A00)(B00 + B01)
    U[1*2+0][5] = 1  # A10
    U[0*2+0][5] = -1 # -A00
    V[0*2+0][5] = 1  # B00
    V[0*2+1][5] = 1  # B01

    # M7 = (A01 - A11)(B10 + B11)
    U[0*2+1][6] = 1  # A01
    U[1*2+1][6] = -1 # -A11
    V[1*2+0][6] = 1  # B10
    V[1*2+1][6] = 1  # B11

    # C[0,0] = M1 + M4 - M5 + M7
    W[0][0] = 1   # M1
    W[0][3] = 1   # M4
    W[0][4] = -1  # -M5
    W[0][6] = 1   # M7

    # C[0,1] = M3 + M5
    W[2][2] = 1   # M3
    W[2][4] = 1   # M5

    # C[1,0] = M2 + M4
    W[1][1] = 1   # M2
    W[1][3] = 1   # M4

    # C[1,1] = M1 - M2 + M3 + M6
    W[3][0] = 1   # M1
    W[3][1] = -1  # -M2
    W[3][2] = 1   # M3
    W[3][5] = 1   # M6

    # Deadline guard: if somehow we've exceeded time, still return the result
    # (this is a constant-time construction, so this should never trigger)
    if time.time() > _deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
