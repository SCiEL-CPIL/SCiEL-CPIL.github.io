import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 15 decomposition
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (1x2) * B1 (2x5)
    # A1 is row 0 of A: indices (0,0), (0,1)
    # B1 is rows 0,1 of B: indices (0,k), (1,k)
    # C row 0
    
    # Term 1: A[0,0] * B[0,k]
    for k in range(p):
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
    col += 1
    
    # Term 2: A[0,1] * B[1,k]
    for k in range(p):
        U[0 * m + 1][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 0][col] = 1
    col += 1
    
    # Part 2: A2 (2x2) * B2a (2x2) using Strassen
    # A2 is rows 1,2 of A: (1,0),(1,1),(2,0),(2,1)
    # B2a is rows 2,3 of B, cols 0,1: (2,0),(2,1),(3,0),(3,1)
    # Output C rows 1,2 cols 0,1
    
    # Strassen M1 = (A11+A22)(B11+B22)
    # A11=A[1,0], A22=A[2,1]
    # B11=B[2,0], B22=B[3,1]
    U[1 * m + 0][col] = 1
    U[2 * m + 1][col] = 1
    V[2 * p + 0][col] = 1
    V[3 * p + 1][col] = 1
    # C11 (Row 1, Col 0) += M1
    W[0 * n + 1][col] = 1
    # C22 (Row 2, Col 1) += M1
    W[1 * n + 2][col] = 1
    col += 1
    
    # Strassen M2 = (A21+A22)B11
    # A21=A[2,0], A22=A[2,1]
    # B11=B[2,0]
    U[2 * m + 0][col] = 1
    U[2 * m + 1][col] = 1
    V[2 * p + 0][col] = 1
    # C21 (Row 2, Col 0) += M2
    W[0 * n + 2][col] = 1
    # C22 (Row 2, Col 1) -= M2
    W[1 * n + 2][col] = -1
    col += 1
    
    # Strassen M3 = A11(B12-B22)
    # A11=A[1,0]
    # B12=B[2,1], B22=B[3,1]
    U[1 * m + 0][col] = 1
    V[2 * p + 1][col] = 1
    V[3 * p + 1][col] = -1
    # C12 (Row 1, Col 1) += M3
    W[1 * n + 1][col] = 1
    # C22 (Row 2, Col 1) += M3
    W[1 * n + 2][col] = 1
    col += 1
    
    # Strassen M4 = A22(B21-B11)
    # A22=A[2,1]
    # B21=B[3,0], B11=B[2,0]
    U[2 * m + 1][col] = 1
    V[3 * p + 0][col] = 1
    V[2 * p + 0][col] = -1
    # C11 (Row 1, Col 0) += M4
    W[0 * n + 1][col] = 1
    # C21 (Row 2, Col 0) += M4
    W[0 * n + 2][col] = 1
    col += 1
    
    # Strassen M5 = (A11+A12)B22
    # A11=A[1,0], A12=A[1,1]
    # B22=B[3,1]
    U[1 * m + 0][col] = 1
    U[1 * m + 1][col] = 1
    V[3 * p + 1][col] = 1
    # C11 (Row 1, Col 0) -= M5
    W[0 * n + 1][col] = -1
    # C12 (Row 1, Col 1) += M5
    W[1 * n + 1][col] = 1
    col += 1
    
    # Strassen M6 = (A21-A11)(B11+B12)
    # A21=A[2,0], A11=A[1,0]
    # B11=B[2,0], B12=B[2,1]
    U[2 * m + 0][col] = 1
    U[1 * m + 0][col] = -1
    V[2 * p + 0][col] = 1
    V[2 * p + 1][col] = 1
    # C22 (Row 2, Col 1) += M6
    W[1 * n + 2][col] = 1
    col += 1
    
    # Strassen M7 = (A12-A22)(B21+B22)
    # A12=A[1,1], A22=A[2,1]
    # B21=B[3,0], B22=B[3,1]
    U[1 * m + 1][col] = 1
    U[2 * m + 1][col] = -1
    V[3 * p + 0][col] = 1
    V[3 * p + 1][col] = 1
    # C11 (Row 1, Col 0) += M7
    W[0 * n + 1][col] = 1
    # C22 (Row 2, Col 1) += M7
    W[1 * n + 2][col] = 1
    col += 1
    
    # Part 3: A2 (2x2) * B2b (2x3) standard
    # B2b is rows 2,3 of B, cols 2,3,4
    for k in range(2, p):
        # Term: A[1,0] * B[2,k] -> C[1,k]
        U[1 * m + 0][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
        # Term: A[1,1] * B[3,k] -> C[1,k]
        U[1 * m + 1][col] = 1
        V[3 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
        # Term: A[2,0] * B[2,k] -> C[2,k]
        U[2 * m + 0][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1
        
        # Term: A[2,1] * B[3,k] -> C[2,k]
        U[2 * m + 1][col] = 1
        V[3 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
