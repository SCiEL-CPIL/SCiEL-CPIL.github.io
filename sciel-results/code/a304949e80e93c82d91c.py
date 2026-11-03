import time
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition for rank 94
    rank = 94
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_laderman(U, V, W, i0, j0, k0, col):
        # Adds 23 terms for 3x3x3 block at (i0, j0, k0)
        # A is 3x3, B is 3x3, C is 3x3
        # Indices: i in i0..i0+2, j in j0..j0+2, k in k0..k0+2
        
        # M1: A00 B00
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        # M2: A00 (B11 + B12 - B21 - B22)
        U[i0 * m + j0][col] += 1
        V[(j0+1) * p + (k0+1)][col] += 1
        V[(j0+1) * p + (k0+2)][col] += 1
        V[(j0+2) * p + (k0+1)][col] -= 1
        V[(j0+2) * p + (k0+2)][col] -= 1
        W[(k0+1) * n + i0][col] += 1
        W[(k0+2) * n + i0][col] += 1
        col += 1
        
        # M3: (A01 + A02) B22
        U[i0 * m + (j0+1)][col] += 1
        U[i0 * m + (j0+2)][col] += 1
        V[(j0+2) * p + (k0+2)][col] += 1
        W[(k0+2) * n + i0][col] += 1
        col += 1
        
        # M4: (A01 - A21) (B10 + B20)
        U[i0 * m + (j0+1)][col] += 1
        U[(i0+2) * m + (j0+1)][col] -= 1
        V[(j0+1) * p + k0][col] += 1
        V[(j0+2) * p + k0][col] += 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + (i0+2)][col] -= 1
        col += 1
        
        # M5: (A10 + A20) (B01 - B00)
        U[(i0+1) * m + j0][col] += 1
        U[(i0+2) * m + j0][col] += 1
        V[j0 * p + (k0+1)][col] += 1
        V[j0 * p + k0][col] -= 1
        W[(k0+1) * n + (i0+1)][col] += 1
        W[(k0+1) * n + (i0+2)][col] += 1
        W[k0 * n + (i0+1)][col] -= 1
        W[k0 * n + (i0+2)][col] -= 1
        col += 1
        
        # M6: (A11 + A22 - A00 - A10 - A20) (B00 + B10 + B20)
        U[(i0+1) * m + (j0+1)][col] += 1
        U[(i0+2) * m + (j0+2)][col] += 1
        U[i0 * m + j0][col] -= 1
        U[(i0+1) * m + j0][col] -= 1
        U[(i0+2) * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[(j0+1) * p + k0][col] += 1
        V[(j0+2) * p + k0][col] += 1
        W[k0 * n + (i0+1)][col] += 1
        W[k0 * n + (i0+2)][col] += 1
        col += 1
        
        # M7: (A02 - A12 + A22) (B11 + B21)
        U[i0 * m + (j0+2)][col] += 1
        U[(i0+1) * m + (j0+2)][col] -= 1
        U[(i0+2) * m + (j0+2)][col] += 1
        V[(j0+1) * p + (k0+1)][col] += 1
        V[(j0+2) * p + (k0+1)][col] += 1
        W[(k0+1) * n + i0][col] += 1
        W[(k0+1) * n + (i0+2)][col] += 1
        col += 1
        
        # M8: (A12 + A22 - A00 - A01 - A02) (B01 + B02 + B12)
        U[(i0+1) * m + (j0+2)][col] += 1
        U[(i0+2) * m + (j0+2)][col] += 1
        U[i0 * m + j0][col] -= 1
        U[i0 * m + (j0+1)][col] -= 1
        U[i0 * m + (j0+2)][col] -= 1
        V[j0 * p + (k0+1)][col] += 1
        V[j0 * p + (k0+2)][col] += 1
        V[(j0+1) * p + (k0+2)][col] += 1
        W[(k0+1) * n + (i0+1)][col] += 1
        W[(k0+1) * n + (i0+2)][col] += 1
        W[(k0+2) * n + (i0+1)][col] += 1
        W[(k0+2) * n + (i0+2)][col] += 1
        col += 1
        
        # M9: (A00 + A11 + A22 - A12 - A21) (B22 + B12 + B21)
        U[i0 * m + j0][col] += 1
        U[(i0+1) * m + (j0+1)][col] += 1
        U[(i0+2) * m + (j0+2)][col] += 1
        U[(i0+1) * m + (j0+2)][col] -= 1
        U[(i0+2) * m + (j0+1)][col] -= 1
        V[(j0+2) * p + (k0+2)][col] += 1
        V[(j0+1) * p + (k0+2)][col] += 1
        V[(j0+2) * p + (k0+1)][col] += 1
        W[(k0+2) * n + i0][col] += 1
        W[(k0+2) * n + (i0+1)][col] += 1
        W[(k0+2) * n + (i0+2)][col] += 1
        col += 1
        
        # M10: (A02 + A12 + A22 - A00 - A01) (B21 + B11 + B10)
        U[i0 * m + (j0+2)][col] += 1
        U[(i0+1) * m + (j0+2)][col] += 1
        U[(i0+2) * m + (j0+2)][col] += 1
        U[i0 * m + j0][col] -= 1
        U[i0 * m + (j0+1)][col] -= 1
        V[(j0+2) * p + (k0+1)][col] += 1
        V[(j0+1) * p + (k0+1)][col] += 1
        V[(j0+1) * p + k0][col] += 1
        W[(k0+1) * n + i0][col] += 1
        W[(k0+1) * n + (i0+1)][col] += 1
        W[(k0+1) * n + (i0+2)][col] += 1
        col += 1
        
        # M11: (A10 + A20 + A21 - A00 - A22) (B02 + B12 + B22)
        U[(i0+1) * m + j0][col] += 1
        U[(i0+2) * m + j0][col] += 1
        U[(i0+2) * m + (j0+1)][col] += 1
        U[i0 * m + j0][col] -= 1
        U[(i0+2) * m + (j0+2)][col] -= 1
        V[j0 * p + (k0+2)][col] += 1
        V[(j0+1) * p + (k0+2)][col] += 1
        V[(j0+2) * p + (k0+2)][col] += 1
        W[(k0+2) * n + (i0+1)][col] += 1
        W[(k0+2) * n + (i0+2)][col] += 1
        col += 1
        
        # M12: (A01 + A11 + A21 - A10 - A20) (B20 + B10 + B00)
        U[i0 * m + (j0+1)][col] += 1
        U[(i0+1) * m + (j0+1)][col] += 1
        U[(i0+2) * m + (j0+1)][col] += 1
        U[(i0+1) * m + j0][col] -= 1
        U[(i0+2) * m + j0][col] -= 1
        V[(j0+2) * p + k0][col] += 1
        V[(j0+1) * p + k0][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + (i0+1)][col] += 1
        W[k0 * n + (i0+2)][col] += 1
        col += 1
        
        # M13: (A01 + A11 + A21 - A10 - A00) (B02 + B22 + B12)
        U[i0 * m + (j0+1)][col] += 1
        U[(i0+1) * m + (j0+1)][col] += 1
        U[(i0+2) * m + (j0+1)][col] += 1
        U[(i0+1) * m + j0][col] -= 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + (k0+2)][col] += 1
        V[(j0+2) * p + (k0+2)][col] += 1
        V[(j0+1) * p + (k0+2)][col] += 1
        W[(k0+2) * n + i0][col] += 1
        W[(k0+2) * n + (i0+1)][col] += 1
        W[(k0+2) * n + (i0+2)][col] += 1
        col += 1
        
        # M14: (A02 + A12 + A22 - A20 - A00) (B21 + B11 + B01)
        U[i0 * m + (j0+2)][col] += 1
        U[(i0+1) * m + (j0+2)][col] += 1
        U[(i0+2) * m + (j0+2)][col] += 1
        U[(i0+2) * m + j0][col] -= 1
        U[i0 * m + j0][col] -= 1
        V[(j0+2) * p + (k0+1)][col] += 1
        V[(j0+1) * p + (k0+1)][col] += 1
        V[j0 * p + (k0+1)][col] += 1
        W[(k0+1) * n + i0][col] += 1
        W[(k0+1) * n + (i0+1)][col] += 1
        W[(k0+1) * n + (i0+2)][col] += 1
        col += 1
        
        # M15: (A10 + A20 + A21 - A00 - A11) (B02 + B12 + B22)
        U[(i0+1) * m + j0][col] += 1
        U[(i0+2) * m + j0][col] += 1
        U[(i0+2) * m + (j0+1)][col] += 1
        U[i0 * m + j0][col] -= 1
        U[(i0+1) * m + (j0+1)][col] -= 1
        V[j0 * p + (k0+2)][col] += 1
        V[(j0+1) * p + (k0+2)][col] += 1
        V[(j0+2) * p + (k0+2)][col] += 1
        W[(k0+2) * n + (i0+1)][col] += 1
        W[(k0+2) * n + (i0+2)][col] += 1
        col += 1
        
        # M16: (A02 + A12 + A22 - A10 - A20) (B21 + B11 + B01)
        U[i0 * m + (j0+2)][col] += 1
        U[(i0+1) * m + (j0+2)][col] += 1
        U[(i0+2) * m + (j0+2)][col] += 1
        U[(i0+1) * m + j0][col] -= 1
        U[(i0+2) * m + j0][col] -= 1
        V[(j0+2) * p + (k0+1)][col] += 1
        V[(j0+1) * p + (k0+1)][col] += 1
        V[j0 * p + (k0+1)][col] += 1
        W[(k0+1) * n + i0][col] += 1
        W[(k0+1) * n + (i0+1)][col] += 1
        W[(k0+1) * n + (i0+2)][col] += 1
        col += 1
        
        # M17: (A01 + A11 + A21 - A00 - A22) (B20 + B10 + B00)
        U[i0 * m + (j0+1)][col] += 1
        U[(i0+1) * m + (j0+1)][col] += 1
        U[(i0+2) * m + (j0+1)][col] += 1
        U[i0 * m + j0][col] -= 1
        U[(i0+2) * m + (j0+2)][col] -= 1
        V[(j0+2) * p + k0][col] += 1
        V[(j0+1) * p + k0][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + (i0+1)][col] += 1
        W[k0 * n + (i0+2)][col] += 1
        col += 1
        
        # M18: (A00 + A11 + A22 - A02 - A20) (B21 + B11 + B01)
        U[i0 * m + j0][col] += 1
        U[(i0+1) * m + (j0+1)][col] += 1
        U[(i0+2) * m + (j0+2)][col] += 1
        U[i0 * m + (j0+2)][col] -= 1
        U[(i0+2) * m + j0][col] -= 1
        V[(j0+2) * p + (k0+1)][col] += 1
        V[(j0+1) * p + (k0+1)][col] += 1
        V[j0 * p + (k0+1)][col] += 1
        W[(k0+1) * n + i0][col] += 1
        W[(k0+1) * n + (i0+1)][col] += 1
        W[(k0+1) * n + (i0+2)][col] += 1
        col += 1
        
        # M19: (A00 + A11 + A22 - A01 - A10) (B20 + B10 + B00)
        U[i0 * m + j0][col] += 1
        U[(i0+1) * m + (j0+1)][col] += 1
        U[(i0+2) * m + (j0+2)][col] += 1
        U[i0 * m + (j0+1)][col] -= 1
        U[(i0+1) * m + j0][col] -= 1
        V[(j0+2) * p + k0][col] += 1
        V[(j0+1) * p + k0][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + (i0+1)][col] += 1
        W[k0 * n + (i0+2)][col] += 1
        col += 1
        
        # M20: (A02 + A12 + A22 - A00 - A11) (B21 + B11 + B01)
        U[i0 * m + (j0+2)][col] += 1
        U[(i0+1) * m + (j0+2)][col] += 1
        U[(i0+2) * m + (j0+2)][col] += 1
        U[i0 * m + j0][col] -= 1
        U[(i0+1) * m + (j0+1)][col] -= 1
        V[(j0+2) * p + (k0+1)][col] += 1
        V[(j0+1) * p + (k0+1)][col] += 1
        V[j0 * p + (k0+1)][col] += 1
        W[(k0+1) * n + i0][col] += 1
        W[(k0+1) * n + (i0+1)][col] += 1
        W[(k0+1) * n + (i0+2)][col] += 1
        col += 1
        
        # M21: (A01 + A11 + A21 - A00 - A20) (B20 + B10 + B00)
        U[i0 * m + (j0+1)][col] += 1
        U[(i0+1) * m + (j0+1)][col] += 1
        U[(i0+2) * m + (j0+1)][col] += 1
        U[i0 * m + j0][col] -= 1
        U[(i0+2) * m + j0][col] -= 1
        V[(j0+2) * p + k0][col] += 1
        V[(j0+1) * p + k0][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + (i0+1)][col] += 1
        W[k0 * n + (i0+2)][col] += 1
        col += 1
        
        # M22: (A00 + A11 + A22 - A02 - A12) (B21 + B11 + B01)
        U[i0 * m + j0][col] += 1
        U[(i0+1) * m + (j0+1)][col] += 1
        U[(i0+2) * m + (j0+2)][col] += 1
        U[i0 * m + (j0+2)][col] -= 1
        U[(i0+1) * m + (j0+2)][col] -= 1
        V[(j0+2) * p + (k0+1)][col] += 1
        V[(j0+1) * p + (k0+1)][col] += 1
        V[j0 * p + (k0+1)][col] += 1
        W[(k0+1) * n + i0][col] += 1
        W[(k0+1) * n + (i0+1)][col] += 1
        W[(k0+1) * n + (i0+2)][col] += 1
        col += 1
        
        # M23: (A00 + A11 + A22 - A01 - A02) (B20 + B10 + B00)
        U[i0 * m + j0][col] += 1
        U[(i0+1) * m + (j0+1)][col] += 1
        U[(i0+2) * m + (j0+2)][col] += 1
        U[i0 * m + (j0+1)][col] -= 1
        U[i0 * m + (j0+2)][col] -= 1
        V[(j0+2) * p + k0][col] += 1
        V[(j0+1) * p + k0][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + (i0+1)][col] += 1
        W[k0 * n + (i0+2)][col] += 1
        col += 1
        
        return col

    def add_strassen(U, V, W, i0, j0, k0, col):
        # Adds 7 terms for 2x2x2 block
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        return col

    def add_2x2x3(U, V, W, i0, j0, k0, col):
        # Adds 11 terms for 2x2x3 block
        # 2x2x1 (4) + 2x2x2 (7)
        # 2x2x1 for k=k0
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        U[i0 * m + j0+1][col] += 1
        V[(j0+1) * p + k0][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        U[(i0+1) * m + j0][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0+1][col] += 1
        col += 1
        U[(i0+1) * m + j0+1][col] += 1
        V[(j0+1) * p + k0][col] += 1
        W[k0 * n + i0+1][col] += 1
        col += 1
        
        # 2x2x2 for k=k0+1, k0+2
        col = add_strassen(U, V, W, i0, j0, k0+1, col)
        return col

    def add_2x2x1(U, V, W, i0, j0, k0, col):
        # Adds 4 terms for 2x2x1 block
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        U[i0 * m + j0+1][col] += 1
        V[(j0+1) * p + k0][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        U[(i0+1) * m + j0][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0+1][col] += 1
        col += 1
        U[(i0+1) * m + j0+1][col] += 1
        V[(j0+1) * p + k0][col] += 1
        W[k0 * n + i0+1][col] += 1
        col += 1
        return col

    def add_1x2x7(U, V, W, i0, j0, k0, col):
        # Adds 14 terms for 1x2x7 block
        # C[i0, k] = A[i0, j0] B[j0, k] + A[i0, j0+1] B[j0+1, k]
        for k in range(7):
            U[i0 * m + j0][col] += 1
            V[j0 * p + (k0+k)][col] += 1
            W[(k0+k) * n + i0][col] += 1
            col += 1
            U[i0 * m + j0+1][col] += 1
            V[(j0+1) * p + (k0+k)][col] += 1
            W[(k0+k) * n + i0][col] += 1
            col += 1
        return col

    # 1. A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    col = add_laderman(U, V, W, 0, 0, 0, col)
    
    # 2. A[0:3, 0:3] * B[0:3, 3:6] -> C[0:3, 3:6]
    col = add_laderman(U, V, W, 0, 0, 3, col)
    
    # 3. A[0:3, 0:3] * B[0:3, 6:7] -> C[0:3, 6:7]
    # 3x3x1: 9 terms
    for i in range(3):
        for j in range(3):
            U[(i) * m + j][col] += 1
            V[j * p + 6][col] += 1
            W[6 * n + i][col] += 1
            col += 1
            
    # 4. A[0:3, 3:5] * B[3:5, 0:7] -> C[0:3, 0:7]
    # Split A into A[0:2, 3:5] and A[2:3, 3:5]
    # 4a. A[0:2, 3:5] * B[3:5, 0:7]
    # 2x2x7: 25 terms
    # Split B into B[3:5, 0:3], B[3:5, 3:6], B[3:5, 6:7]
    # 2x2x3: 11 terms
    col = add_2x2x3(U, V, W, 0, 3, 0, col)
    # 2x2x3: 11 terms
    col = add_2x2x3(U, V, W, 0, 3, 3, col)
    # 2x2x1: 4 terms
    col = add_2x2x1(U, V, W, 0, 3, 6, col)
    
    # 4b. A[2:3, 3:5] * B[3:5, 0:7]
    # 1x2x7: 14 terms
    col = add_1x2x7(U, V, W, 2, 3, 0, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
