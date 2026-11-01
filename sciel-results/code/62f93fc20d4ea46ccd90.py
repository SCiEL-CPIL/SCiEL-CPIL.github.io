"""
Rank-minimal exact tensor decomposition for 2x2 matrix multiplication.

The known optimal rank for 2x2 matrix multiplication is 7 (Strassen's algorithm).
This program returns the verified rank-7 Strassen decomposition.

A rank-6 solution is known to be impossible (the tensor rank of the 2x2 matmul
tensor is exactly 7).  Therefore the optimal score is rank 7.
"""
import numpy as np


def construct():
    """
    Return a rank-7 exact tensor decomposition for 2x2 matrix multiplication
    using Strassen's algorithm.

    Returns a dict with keys 'U', 'V', 'W' where:
      U: (n*m, rank) = (4, 7)  -- factors for A
      V: (m*p, rank) = (4, 7)  -- factors for B
      W: (p*n, rank) = (4, 7)  -- factors for C, with W[k*n+i, r] convention

    The decomposition satisfies:
      (A * B)[i, k] = sum_r  (sum_a U[i*m+a, r] * V[a*p+k, r]) * W[k*n+i, r]
    for all i, k in {0, 1}.
    """
    n, m, p = 2, 2, 2
    rank = 7

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen's 7 products:
    #   M1 = (A00 + A11) * (B00 + B11)
    #   M2 = (A10 + A11) * B00
    #   M3 = A00 * (B01 - B11)
    #   M4 = A11 * (B10 - B00)
    #   M5 = (A00 + A01) * B11
    #   M6 = (A10 - A00) * (B00 + B01)
    #   M7 = (A01 - A11) * (B10 + B11)
    #
    # Output:
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # Indexing:
    #   U[i*m + a, r]  -- A part, i=row, a=col
    #   V[a*p + k, r]  -- B part, a=row, k=col
    #   W[k*n + i, r]  -- C part, k=col, i=row  (official trace convention)

    # M1: (A00 + A11)(B00 + B11)
    U[0, 0] = 1   # A00
    U[3, 0] = 1   # A11
    V[0, 0] = 1   # B00
    V[3, 0] = 1   # B11

    # M2: (A10 + A11) B00
    U[2, 1] = 1   # A10
    U[3, 1] = 1   # A11
    V[0, 1] = 1   # B00

    # M3: A00 (B01 - B11)
    U[0, 2] = 1   # A00
    V[1, 2] = 1   # B01
    V[3, 2] = -1  # -B11

    # M4: A11 (B10 - B00)
    U[3, 3] = 1   # A11
    V[2, 3] = 1   # B10
    V[0, 3] = -1  # -B00

    # M5: (A00 + A01) B11
    U[0, 4] = 1   # A00
    U[1, 4] = 1   # A01
    V[3, 4] = 1   # B11

    # M6: (A10 - A00)(B00 + B01)
    U[2, 5] = 1   # A10
    U[0, 5] = -1  # -A00
    V[0, 5] = 1   # B00
    V[1, 5] = 1   # B01

    # M7: (A01 - A11)(B10 + B11)
    U[1, 6] = 1   # A01
    U[3, 6] = -1  # -A11
    V[2, 6] = 1   # B10
    V[3, 6] = 1   # B11

    # W coefficients: W[k*n + i, r]
    # C[0,0]: k=0, i=0 -> W[0, r]
    W[0, 0] = 1   # M1
    W[0, 3] = 1   # M4
    W[0, 4] = -1  # -M5
    W[0, 6] = 1   # M7

    # C[0,1]: k=1, i=0 -> W[2, r]
    W[2, 2] = 1   # M3
    W[2, 4] = 1   # M5

    # C[1,0]: k=0, i=1 -> W[1, r]
    W[1, 1] = 1   # M2
    W[1, 3] = 1   # M4

    # C[1,1]: k=1, i=1 -> W[3, r]
    W[3, 0] = 1   # M1
    W[3, 1] = -1  # -M2
    W[3, 2] = 1   # M3
    W[3, 5] = 1   # M6

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
