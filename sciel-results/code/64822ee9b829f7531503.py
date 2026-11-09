import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 6 Strassen blocks of 2x2x2 (rank 7 each) -> 42
    # 1 Schoolbook block of 2x1x6 (rank 12) -> 12
    # Total rank = 54
    rank = 54
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(j1, j2, k1, k2):
        nonlocal col
        # A indices: (i, j) -> i*m + j
        # B indices: (j, k) -> j*p + k
        # C indices: (i, k) -> k*n + i
        
        # A matrix local 2x2:
        # A[0][0] at (0, j1), A[0][1] at (0, j2)
        # A[1][0] at (1, j1), A[1][1] at (1, j2)
        a00 = 0 * m + j1
        a01 = 0 * m + j2
        a10 = 1 * m + j1
        a11 = 1 * m + j2
        
        # B matrix local 2x2:
        # B[0][0] at (j1, k1), B[0][1] at (j1, k2)
        # B[1][0] at (j2, k1), B[1][1] at (j2, k2)
        b00 = j1 * p + k1
        b01 = j1 * p + k2
        b10 = j2 * p + k1
        b11 = j2 * p + k2
        
        # C matrix local 2x2:
        # C[0][0] at (0, k1) -> k1*n + 0
        # C[0][1] at (0, k2) -> k2*n + 0
        # C[1][0] at (1, k1) -> k1*n + 1
        # C[1][1] at (1, k2) -> k2*n + 1
        c00 = k1 * n + 0
        c01 = k2 * n + 0
        c10 = k1 * n + 1
        c11 = k2 * n + 1
        
        # Strassen products M1..M7
        # M1 = (A00 + A11)(B00 + B11)
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        W[c01][col] -= 1
        W[c10][col] -= 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M3 = A00(B01 - B11)
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M4 = A11(B10 - B00)
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c10][col] -= 1
        col += 1

    # Block 1: J in {0, 1}
    # Split K into {0,1}, {2,3}, {4,5}
    add_strassen(0, 1, 0, 1)
    add_strassen(0, 1, 2, 3)
    add_strassen(0, 1, 4, 5)
    
    # Block 2: J in {2, 3}
    # Split K into {0,1}, {2,3}, {4,5}
    add_strassen(2, 3, 0, 1)
    add_strassen(2, 3, 2, 3)
    add_strassen(2, 3, 4, 5)
    
    # Block 3: J in {4}
    # This is 2x1x6 schoolbook
    # A col 4, B row 4
    # C is 2x6
    for i in range(n):
        for k in range(p):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
