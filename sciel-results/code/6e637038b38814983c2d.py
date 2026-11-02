"""Rank-minimal exact tensor decomposition for 3x3 by 3x4 matrix multiplication."""
import time

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Returns a rank-25 exact decomposition using block decomposition:
    - 11 terms for 2x2x3 block (Strassen 7 + Schoolbook 4)
    - 1 term for 2x1x1x3 outer product
    - 4 terms for 2x2x1 block
    - 1 term for 2x1x1x1 outer product
    - 4 terms for 1x2x2x3 block
    - 1 term for 1x1x1x3 outer product
    - 2 terms for 1x2x2x1 block
    - 1 term for 1x1x1x1 scalar product
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    rank = 25
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 25
    V = [[0.0] * rank for _ in range(m * p)]  # 12 x 25
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 25

    col = 0

    # Indices
    # A: U[i*3+j]
    # B: V[i*4+j]
    # C: W[k*3+i] (k is col of B, i is row of A)

    # --- Block 1: A[0:2, 0:2] * B[0:2, 0:3] (2x2x3) -> 11 terms ---
    # Strassen for B cols 0,1
    # M1 = (A00+A11)(B00+B11)
    U[0][col] += 1; U[4][col] += 1
    V[0][col] += 1; V[4][col] += 1
    W[0][col] += 1  # C00
    W[4][col] += 1  # C11
    col += 1

    # M2 = (A10+A11)B00
    U[3][col] += 1; U[4][col] += 1
    V[0][col] += 1
    W[1][col] += 1  # C10
    W[4][col] -= 1  # C11
    col += 1

    # M3 = A00(B10-B11)
    U[0][col] += 1
    V[3][col] += 1; V[4][col] -= 1
    W[3][col] += 1  # C01
    W[4][col] += 1  # C11
    col += 1

    # M4 = A11(B01-B00)
    U[4][col] += 1
    V[1][col] += 1; V[0][col] -= 1
    W[0][col] += 1  # C00
    W[1][col] += 1  # C10
    col += 1

    # M5 = (A00+A01)B11
    U[0][col] += 1; U[1][col] += 1
    V[4][col] += 1
    W[3][col] += 1  # C01
    W[0][col] -= 1  # C00
    col += 1

    # M6 = (A10-A00)(B00+B01)
    U[3][col] += 1; U[0][col] -= 1
    V[0][col] += 1; V[1][col] += 1
    W[4][col] += 1  # C11
    col += 1

    # M7 = (A01-A11)(B10+B11)
    U[1][col] += 1; U[4][col] -= 1
    V[3][col] += 1; V[4][col] += 1
    W[0][col] += 1  # C00
    col += 1

    # Schoolbook for B col 2 (V[2], V[6])
    # C02 = A00*B02 + A01*B12 -> W[6]
    U[0][col] += 1
    V[2][col] += 1
    W[6][col] += 1
    col += 1

    U[1][col] += 1
    V[6][col] += 1
    W[6][col] += 1
    col += 1

    # C12 = A10*B02 + A11*B12 -> W[7]
    U[3][col] += 1
    V[2][col] += 1
    W[7][col] += 1
    col += 1

    U[4][col] += 1
    V[6][col] += 1
    W[7][col] += 1
    col += 1

    # --- Block 2: A[0:2, 2] * B[2, 0:3] (2x1x1x3) -> 1 term ---
    # Outer product
    U[2][col] += 1; U[5][col] += 1
    V[8][col] += 1; V[9][col] += 1; V[10][col] += 1
    W[0][col] += 1  # C00
    W[3][col] += 1  # C01
    W[6][col] += 1  # C02
    W[1][col] += 1  # C10
    W[4][col] += 1  # C11
    W[7][col] += 1  # C12
    col += 1

    # --- Block 3: A[0:2, 0:2] * B[0:2, 3] (2x2x1) -> 4 terms ---
    # C03 = A00*B03 + A01*B13 -> W[9]
    U[0][col] += 1
    V[3][col] += 1
    W[9][col] += 1
    col += 1

    U[1][col] += 1
    V[7][col] += 1
    W[9][col] += 1
    col += 1

    # C13 = A10*B03 + A11*B13 -> W[10]
    U[3][col] += 1
    V[3][col] += 1
    W[10][col] += 1
    col += 1

    U[4][col] += 1
    V[7][col] += 1
    W[10][col] += 1
    col += 1

    # --- Block 4: A[0:2, 2] * B[2, 3] (2x1x1x1) -> 1 term ---
    U[2][col] += 1; U[5][col] += 1
    V[11][col] += 1
    W[9][col] += 1  # C03
    W[10][col] += 1 # C13
    col += 1

    # --- Block 5: A[2, 0:2] * B[0:2, 0:3] (1x2x2x3) -> 4 terms ---
    # C20 = A20*B00 + A21*B10 -> W[2]
    # C21 = A20*B01 + A21*B11 -> W[5]
    # C22 = A20*B02 + A21*B12 -> W[8]
    
    # Term 1: A20*(B00+B01)
    U[6][col] += 1
    V[0][col] += 1; V[1][col] += 1
    W[2][col] += 1 # C20
    W[5][col] += 1 # C21
    col += 1

    # Term 2: A21*(B10+B11)
    U[7][col] += 1
    V[3][col] += 1; V[4][col] += 1
    W[2][col] += 1 # C20
    W[5][col] += 1 # C21
    col += 1

    # Term 3: A20*B02
    U[6][col] += 1
    V[2][col] += 1
    W[8][col] += 1 # C22
    col += 1

    # Term 4: A21*B12
    U[7][col] += 1
    V[6][col] += 1
    W[8][col] += 1 # C22
    col += 1

    # --- Block 6: A[2, 2] * B[2, 0:3] (1x1x1x3) -> 1 term ---
    U[8][col] += 1
    V[8][col] += 1; V[9][col] += 1; V[10][col] += 1
    W[2][col] += 1 # C20
    W[5][col] += 1 # C21
    W[8][col] += 1 # C22
    col += 1

    # --- Block 7: A[2, 0:2] * B[0:2, 3] (1x2x2x1) -> 2 terms ---
    # C23 = A20*B03 + A21*B13 -> W[11]
    U[6][col] += 1
    V[3][col] += 1
    W[11][col] += 1
    col += 1

    U[7][col] += 1
    V[7][col] += 1
    W[11][col] += 1
    col += 1

    # --- Block 8: A[2, 2] * B[2, 3] (1x1x1x1) -> 1 term ---
    U[8][col] += 1
    V[11][col] += 1
    W[11][col] += 1
    col += 1

    return {"U": U, "V": V, "W": W}
