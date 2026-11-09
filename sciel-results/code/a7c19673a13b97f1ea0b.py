import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    # Initialize U, V, W
    # U: (n*m, rank) = (6, 17)
    # V: (m*p, rank) = (9, 17)
    # W: (p*n, rank) = (6, 17)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: C_0 = A * B_0 (2x3 * 3x1)
    # A indices: 0,1,2 (row 0), 3,4,5 (row 1)
    # B_0 indices: 0, 3, 6 (col 0 of B)
    # C_0 indices: 0 (C_00), 1 (C_10)
    
    # Terms: A_00*B_00, A_01*B_10, A_02*B_20, A_10*B_00, A_11*B_10, A_12*B_20
    # 1. A_00 * B_00 -> C_00
    U[0][col] = 1; V[0][col] = 1; W[0][col] = 1; col += 1
    # 2. A_01 * B_10 -> C_00
    U[1][col] = 1; V[3][col] = 1; W[0][col] = 1; col += 1
    # 3. A_02 * B_20 -> C_00
    U[2][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
    # 4. A_10 * B_00 -> C_10
    U[3][col] = 1; V[0][col] = 1; W[1][col] = 1; col += 1
    # 5. A_11 * B_10 -> C_10
    U[4][col] = 1; V[3][col] = 1; W[1][col] = 1; col += 1
    # 6. A_12 * B_20 -> C_10
    U[5][col] = 1; V[6][col] = 1; W[1][col] = 1; col += 1
    
    # Part 2: C_1, C_2 = A * B_1:3 (2x3 * 3x2)
    # Split B into B_row0 (1x2) and B_rows12 (2x2)
    # B_row0 indices: 0, 1
    # B_rows12 indices: 3,4,6,7
    
    # Sub-part 2a: A_col0 * B_row0 (2x1 * 1x2)
    # A_col0: 0, 3
    # B_row0: 0, 1
    # Contrib to C_01 (idx 2), C_11 (idx 3), C_02 (idx 4), C_12 (idx 5)
    
    # 7. A_00 * B_00 -> C_01
    U[0][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
    # 8. A_00 * B_01 -> C_02
    U[0][col] = 1; V[1][col] = 1; W[4][col] = 1; col += 1
    # 9. A_10 * B_00 -> C_11
    U[3][col] = 1; V[0][col] = 1; W[3][col] = 1; col += 1
    # 10. A_10 * B_01 -> C_12
    U[3][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
    
    # Sub-part 2b: A_rows12 * B_rows12 (2x2 * 2x2)
    # A_rows12: 1, 2 (row 0), 4, 5 (row 1)
    # a=A_01(1), b=A_02(2), c=A_11(4), d=A_12(5)
    # B_rows12: 3,4 (row 1), 6,7 (row 2)
    # e=B_10(3), f=B_11(4), g=B_20(6), h=B_21(7)
    # Contrib to C_01(2), C_02(4), C_11(3), C_12(5)
    
    # M1 = (a+d)(e+h)
    # a+d: A_01, A_12
    # e+h: B_10, B_21
    U[1][col] = 1; U[5][col] = 1
    V[3][col] = 1; V[7][col] = 1
    W[2][col] = 1  # C_01
    W[5][col] = 1  # C_12
    col += 1
    
    # M2 = (c+d)e
    # c+d: A_11, A_12
    # e: B_10
    U[4][col] = 1; U[5][col] = 1
    V[3][col] = 1
    W[3][col] = 1  # C_11
    W[5][col] = -1 # C_12
    col += 1
    
    # M3 = a(f-h)
    # a: A_01
    # f-h: B_11, -B_21
    U[1][col] = 1
    V[4][col] = 1; V[7][col] = -1
    W[4][col] = 1  # C_02
    W[5][col] = 1  # C_12
    col += 1
    
    # M4 = d(g-e)
    # d: A_12
    # g-e: B_20, -B_10
    U[5][col] = 1
    V[6][col] = 1; V[3][col] = -1
    W[2][col] = 1  # C_01
    W[3][col] = 1  # C_11
    col += 1
    
    # M5 = (a+b)h
    # a+b: A_01, A_02
    # h: B_21
    U[1][col] = 1; U[2][col] = 1
    V[7][col] = 1
    W[2][col] = -1 # C_01
    W[4][col] = 1  # C_02
    col += 1
    
    # M6 = (c-d)(f+h)
    # c-d: A_11, -A_12
    # f+h: B_11, B_21
    U[4][col] = 1; U[5][col] = -1
    V[4][col] = 1; V[7][col] = 1
    W[5][col] = 1  # C_12
    col += 1
    
    # M7 = (b-c)(e+g)
    # b-c: A_02, -A_11
    # e+g: B_10, B_20
    U[2][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[6][col] = 1
    W[2][col] = 1  # C_01
    col += 1
    
    return {"U": U, "V": V, "W": W}
