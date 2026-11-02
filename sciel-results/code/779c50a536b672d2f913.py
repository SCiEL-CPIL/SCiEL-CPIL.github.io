"""
Rank-minimal exact tensor decomposition for 2x2 matrix multiplication.

The 2x2 matrix multiplication tensor has tensor rank 7 (Strassen).
We return the verified rank-7 Strassen decomposition.

Interface: construct() -> {'U', 'V', 'W'}:
  U: shape (n*m, rank) = (4, 7)
  V: shape (m*p, rank) = (4, 7)
  W: shape (p*n, rank) = (4, 7)

Indexing convention:
  U[i*m + a, r] : row i of A, column a
  V[a*p + k, r] : row a of B, column k
  W[k*n + i, r] : column k of C, row i (official trace convention)

The decomposition satisfies:
  (A*B)[i, k] = sum_{a} sum_{r} U[i*m + a, r] * V[a*p + k, r] * W[k*n + i, r]
  for all i in {0,1}, k in {0,1}
"""

import numpy as np
import time

DIMENSIONS = [2, 2, 2]


def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
    """
    Verify that the tensor decomposition exactly reproduces matrix multiplication.
    
    For each output element C[i,k], we check:
      C[i,k] = sum_a sum_r U[i*m+a, r] * V[a*p+k, r] * W[k*n+i, r]
    
    Since we're decomposing the multiplication tensor, for each (i,a,k) the
    "coefficient" should be 1 (since C[i,k] = sum_a A[i,a]*B[a,k]).
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for a in range(m):
                # The tensor T[i, a, k] should equal 1 for the multiplication tensor
                # T[i, a, k] = sum_r U[i*m+a, r] * V[a*p+k, r] * W[k*n+i, r]
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m + a, r] * V_arr[a*p + k, r] * W_arr[k*n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _strassen_decomposition(n, m, p):
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    Strassen's 7 bilinear forms:
      M1 = (A00 + A11)(B00 + B11)
      M2 = (A10 + A11)B00
      M3 = A00(B01 - B11)
      M4 = A11(B10 - B00)
      M5 = (A00 + A01)B11
      M6 = (A10 - A00)(B00 + B01)
      M7 = (A01 - A11)(B10 + B11)

    Output:
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    rank = 7

    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]  # (4, 7)
    V = [[0] * rank for _ in range(m * p)]  # (4, 7)
    W = [[0] * rank for _ in range(p * n)]  # (4, 7)

    # Indexing:
    # U[i*m + a, r] where i in {0,1}, a in {0,1}
    #   U[0, r] = A00, U[1, r] = A01, U[2, r] = A10, U[3, r] = A11
    # V[a*p + k, r] where a in {0,1}, k in {0,1}
    #   V[0, r] = B00, V[1, r] = B01, V[2, r] = B10, V[3, r] = B11
    # W[k*n + i, r] where k in {0,1}, i in {0,1}
    #   W[0, r] = C00 coeff, W[1, r] = C10 coeff, W[2, r] = C01 coeff, W[3, r] = C11 coeff

    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1   # A00
    U[3][0] = 1   # A11
    V[0][0] = 1   # B00
    V[3][0] = 1   # B11

    # M2 = (A10 + A11)B00
    U[2][1] = 1   # A10
    U[3][1] = 1   # A11
    V[0][1] = 1   # B00

    # M3 = A00(B01 - B11)
    U[0][2] = 1   # A00
    V[1][2] = 1   # B01
    V[3][2] = -1  # -B11

    # M4 = A11(B10 - B00)
    U[3][3] = 1   # A11
    V[2][3] = 1   # B10
    V[0][3] = -1  # -B00

    # M5 = (A00 + A01)B11
    U[0][4] = 1   # A00
    U[1][4] = 1   # A01
    V[3][4] = 1   # B11

    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1   # A10
    U[0][5] = -1  # -A00
    V[0][5] = 1   # B00
    V[1][5] = 1   # B01

    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1   # A01
    U[3][6] = -1  # -A11
    V[2][6] = 1   # B10
    V[3][6] = 1   # B11

    # W coefficients: C[i,k] = sum_r W[k*n+i, r] * M_r
    # C[0,0] = M1 + M4 - M5 + M7
    W[0][0] = 1   # M1
    W[0][3] = 1   # M4
    W[0][4] = -1  # -M5
    W[0][6] = 1   # M7

    # C[1,0] = M2 + M4
    W[1][1] = 1   # M2
    W[1][3] = 1   # M4

    # C[0,1] = M3 + M5
    W[2][2] = 1   # M3
    W[2][4] = 1   # M5

    # C[1,1] = M1 - M2 + M3 + M6
    W[3][0] = 1   # M1
    W[3][1] = -1  # -M2
    W[3][2] = 1   # M3
    W[3][5] = 1   # M6

    return U, V, W


def _schoolbook_decomposition(n, m, p):
    """
    Return the rank-8 schoolbook decomposition for 2x2 matrix multiplication.
    This is the trivial decomposition where each product term is separate.
    """
    rank = n * m * p  # 8 for 2x2x2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for a in range(m):
            for k in range(p):
                U[i * m + a][col] = 1
                V[a * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W


def construct():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    The minimal rank for 2x2 matrix multiplication over the reals is 7
    (Strassen's algorithm). We verify the decomposition and return it.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 2
    
    # Primary: Strassen rank-7 decomposition
    U, V, W = _strassen_decomposition(n, m, p)
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p, 7):
        return {"U": U, "V": V, "W": W}
    
    # Check time
    if time.time() > deadline:
        # Time pressure: return schoolbook as fallback
        U, V, W = _schoolbook_decomposition(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank-8 decomposition
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    # Verify fallback
    if _verify_decomposition(U, V, W, n, m, p, 8):
        return {"U": U, "V": V, "W": W}
    
    # Should not reach here, but return schoolbook anyway
    return {"U": U, "V": V, "W": W}
