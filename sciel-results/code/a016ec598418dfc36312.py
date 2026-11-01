"""
Tensor decomposition for 2x2 matrix multiplication.

This program returns the rank-7 Strassen decomposition for multiplying
two 2x2 matrices. The rank 7 is the theoretical minimum for this problem
over fields of characteristic 0.

Interface:
    construct() -> {'U', 'V', 'W'}
    U: shape (4, 7)
    V: shape (4, 7)
    W: shape (4, 7)
    
    Indices follow the convention:
    U[i*m + j, r] corresponds to A[i, j]
    V[j*p + k, r] corresponds to B[j, k]
    W[k*n + i, r] corresponds to C[i, k]
    where n=m=p=2.
"""

def construct():
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices
    # U maps A elements to Strassen products
    # V maps B elements to Strassen products
    # W maps Strassen products to C elements
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products:
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B01)
    # M6 = (A01 - A11) * (B10 + B11)
    
    # Reconstruction:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    # --- Product 0: M0 ---
    # A: A00 (idx 0), A11 (idx 3)
    U[0][0] = 1
    U[3][0] = 1
    # B: B00 (idx 0), B11 (idx 3)
    V[0][0] = 1
    V[3][0] = 1
    # C: C00 (idx 0), C11 (idx 3)
    W[0][0] = 1
    W[3][0] = 1

    # --- Product 1: M1 ---
    # A: A10 (idx 2), A11 (idx 3)
    U[2][1] = 1
    U[3][1] = 1
    # B: B00 (idx 0)
    V[0][1] = 1
    # C: C10 (idx 1), C11 (idx 3)
    W[1][1] = 1
    W[3][1] = -1

    # --- Product 2: M2 ---
    # A: A00 (idx 0)
    U[0][2] = 1
    # B: B01 (idx 1), B11 (idx 3)
    V[1][2] = 1
    V[3][2] = -1
    # C: C01 (idx 2), C11 (idx 3)
    W[2][2] = 1
    W[3][2] = 1

    # --- Product 3: M3 ---
    # A: A11 (idx 3)
    U[3][3] = 1
    # B: B10 (idx 2), B00 (idx 0)
    V[2][3] = 1
    V[0][3] = -1
    # C: C00 (idx 0), C10 (idx 1)
    W[0][3] = 1
    W[1][3] = 1

    # --- Product 4: M4 ---
    # A: A00 (idx 0), A01 (idx 1)
    U[0][4] = 1
    U[1][4] = 1
    # B: B11 (idx 3)
    V[3][4] = 1
    # C: C00 (idx 0), C01 (idx 2)
    W[0][4] = -1
    W[2][4] = 1

    # --- Product 5: M5 ---
    # A: A10 (idx 2), A00 (idx 0)
    U[2][5] = 1
    U[0][5] = -1
    # B: B00 (idx 0), B01 (idx 1)
    V[0][5] = 1
    V[1][5] = 1
    # C: C11 (idx 3)
    W[3][5] = 1

    # --- Product 6: M6 ---
    # A: A01 (idx 1), A11 (idx 3)
    U[1][6] = 1
    U[3][6] = -1
    # B: B10 (idx 2), B11 (idx 3)
    V[2][6] = 1
    V[3][6] = 1
    # C: C00 (idx 0)
    W[0][6] = 1

    return {"U": U, "V": V, "W": W}
