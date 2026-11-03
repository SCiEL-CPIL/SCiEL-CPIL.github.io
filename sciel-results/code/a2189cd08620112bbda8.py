import time
import sys

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    # Part 1: 2x4 * 4x4 using 4 Strassen 2x2 multiplications (rank 28)
    # Part 2: 1x4 * 4x4 using naive (rank 16)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen term
    # A is 2x2 block of A (rows i1,i2, cols j1,j2)
    # B is 2x2 block of B (rows j1,j2, cols k1,k2)
    # Output C is 2x2 block (rows i1,i2, cols k1,k2)
    def add_strassen(i1, i2, j1, j2, k1, k2, offset):
        nonlocal col
        # Strassen coefficients for A
        # M1: (a11+a22)(b11+b22)
        # M2: (a21+a22)b11
        # M3: a11(b12-b22)
        # M4: a22(b21-b11)
        # M5: (a11+a12)b22
        # M6: (a21-a11)(b11+b12)
        # M7: (a12-a22)(b21+b22)
        
        # A indices in U: i*m + j
        # B indices in V: j*p + k
        # C indices in W: k*n + i
        
        # We need to add 7 columns.
        
        # Term 1
        # A: a11, a22
        # B: b11, b22
        # C: c11, c12, c21, c22
        U[i1*m + j1][col] = 1
        U[i2*m + j2][col] = 1
        V[j1*p + k1][col] = 1
        V[j2*p + k2][col] = 1
        W[k1*n + i1][col] = 1
        W[k2*n + i2][col] = 1
        W[k2*n + i1][col] = 1
        W[k1*n + i2][col] = 1
        col += 1
        
        # Term 2
        # A: a21, a22
        # B: b11
        # C: c21, c22
        U[i2*m + j1][col] = 1
        U[i2*m + j2][col] = 1
        V[j1*p + k1][col] = 1
        W[k1*n + i2][col] = 1
        W[k2*n + i2][col] = 1
        col += 1
        
        # Term 3
        # A: a11
        # B: b12, b22
        # C: c12, c22
        U[i1*m + j1][col] = 1
        V[j1*p + k2][col] = 1
        V[j2*p + k2][col] = -1
        W[k2*n + i1][col] = 1
        W[k2*n + i2][col] = 1
        col += 1
        
        # Term 4
        # A: a22
        # B: b21, b11
        # C: c21, c22
        U[i2*m + j2][col] = 1
        V[j2*p + k1][col] = 1
        V[j1*p + k1][col] = -1
        W[k1*n + i2][col] = 1
        W[k2*n + i2][col] = 1
        col += 1
        
        # Term 5
        # A: a11, a12
        # B: b22
        # C: c12, c22
        U[i1*m + j1][col] = 1
        U[i1*m + j2][col] = 1
        V[j2*p + k2][col] = 1
        W[k2*n + i1][col] = 1
        W[k2*n + i2][col] = 1
        col += 1
        
        # Term 6
        # A: a21, a11
        # B: b11, b12
        # C: c21, c22
        U[i2*m + j1][col] = 1
        U[i1*m + j1][col] = -1
        V[j1*p + k1][col] = 1
        V[j1*p + k2][col] = 1
        W[k1*n + i2][col] = 1
        W[k2*n + i2][col] = 1
        col += 1
        
        # Term 7
        # A: a12, a22
        # B: b21, b22
        # C: c11, c12
        U[i1*m + j2][col] = 1
        U[i2*m + j2][col] = -1
        V[j2*p + k1][col] = 1
        V[j2*p + k2][col] = 1
        W[k1*n + i1][col] = 1
        W[k2*n + i1][col] = 1
        col += 1

    # Apply Strassen to the 4 blocks of the 2x4 * 4x4 product
    # A rows 0,1. B rows/cols 0,1 -> C rows 0,1, cols 0,1
    add_strassen(0, 1, 0, 1, 0, 1, 0)
    # A rows 0,1. B rows/cols 2,3? No.
    # A is 2x4. B is 4x4.
    # A = [A11 A12] where A11 is cols 0,1 and A12 is cols 2,3.
    # B = [[B11 B12], [B21 B22]] where B11 is rows 0,1 cols 0,1.
    
    # Product C = [ A11*B11 + A12*B21 , A11*B12 + A12*B22 ]
    
    # Block 1: A11 (rows 0,1, cols 0,1) * B11 (rows 0,1, cols 0,1) -> C part (rows 0,1, cols 0,1)
    add_strassen(0, 1, 0, 1, 0, 1, 0)
    
    # Block 2: A12 (rows 0,1, cols 2,3) * B21 (rows 2,3, cols 0,1) -> C part (rows 0,1, cols 0,1)
    add_strassen(0, 1, 2, 3, 0, 1, 0)
    
    # Block 3: A11 (rows 0,1, cols 0,1) * B12 (rows 0,1, cols 2,3) -> C part (rows 0,1, cols 2,3)
    add_strassen(0, 1, 0, 1, 2, 3, 0)
    
    # Block 4: A12 (rows 0,1, cols 2,3) * B22 (rows 2,3, cols 2,3) -> C part (rows 0,1, cols 2,3)
    add_strassen(0, 1, 2, 3, 2, 3, 0)
    
    # Part 2: 1x4 * 4x4
    # A row 2 (index 2). B all. C row 2.
    for j in range(4):
        for k in range(4):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
