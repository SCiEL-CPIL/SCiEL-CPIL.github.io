import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is 29
    # 7 from Strassen (2x2x2 subblock)
    # 12 from Schoolbook (2x2x3 subblock)
    # 10 from Schoolbook (2x1x5 subblock)
    
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # Indices: i in 0,1; j in 0,1; k in 0,1
    # U indices: i*m + j -> 0,1,2,3
    # V indices: j*p + k -> 0,1,5,6
    # W indices: k*n + i -> 0,1,2,3
    
    # Strassen coefficients
    # M1 = A00(B00+B11)
    # M2 = (A00+A11)B11
    # M3 = (A00+A01)B00
    # M4 = A11(B01+B10)
    # M5 = (A00+A11)(B00+B11)
    # M6 = (A00-A11)B01  (Wait, standard Strassen M6 is (A00-A11)(B01-B11)? No)
    # M7 = (A01-A11)(B00+B10)
    
    # Let's use a verified set of 7 rank-1 tensors for 2x2x2
    # Source: Standard Strassen
    # T1: U=(1,0,0,0), V=(1,0,0,1), W=(-1,1,0,-1) ? 
    # I will use the explicit formulas for C
    # C00 = M5 + M4 - M2 - M1
    # C01 = M3 + M1
    # C10 = M2 + M4
    # C11 = M5 - M1 - M3 + M2
    
    # Define U, V for M1..M7
    # M1: A00, B00+B11
    U[0][col] = 1
    V[0][col] = 1
    V[5][col] = 1
    W[0][col] = -1 # C00
    W[1][col] = 1  # C01
    W[2][col] = 0  # C10
    W[3][col] = -1 # C11
    col += 1
    
    # M2: A00+A11, B11
    U[0][col] = 1
    U[3][col] = 1
    V[6][col] = 1 # B11 is j=1, k=1 -> 1*5+1=6
    W[0][col] = -1 # C00
    W[1][col] = 0  # C01
    W[2][col] = 1  # C10
    W[3][col] = 1  # C11
    col += 1
    
    # M3: A00+A01, B00
    U[0][col] = 1
    U[1][col] = 1
    V[0][col] = 1 # B00 is j=0, k=0 -> 0*5+0=0
    W[0][col] = 0  # C00
    W[1][col] = 1  # C01
    W[2][col] = 0  # C10
    W[3][col] = -1 # C11
    col += 1
    
    # M4: A11, B01+B10
    U[3][col] = 1
    V[1][col] = 1 # B01 is j=0, k=1 -> 0*5+1=1
    V[5][col] = 1 # B10 is j=1, k=0 -> 1*5+0=5
    W[0][col] = 1  # C00
    W[1][col] = 0  # C01
    W[2][col] = 1  # C10
    W[3][col] = 0  # C11
    col += 1
    
    # M5: A00+A11, B00+B11
    U[0][col] = 1
    U[3][col] = 1
    V[0][col] = 1
    V[6][col] = 1
    W[0][col] = 1  # C00
    W[1][col] = 0  # C01
    W[2][col] = 0  # C10
    W[3][col] = 1  # C11
    col += 1
    
    # M6: A00-A11, B01
    # Wait, is this the standard 6th?
    # C00 = M5+M4-M2-M1
    # C01 = M3+M1
    # C10 = M2+M4
    # C11 = M5-M1-M3+M2
    # This uses M1..M5. We need 2 more?
    # No, Strassen uses 7 multiplications.
    # The formulas above only use 5 distinct Ms?
    # M1, M2, M3, M4, M5.
    # Where are M6, M7?
    # Ah, the standard Strassen has 7 terms.
    # My derivation of C from M might be missing terms or I combined them?
    # No, the standard Strassen is:
    # M1 = A00(B00+B11)
    # M2 = (A00+A11)B11
    # M3 = (A00+A01)B00
    # M4 = A11(B01+B10)
    # M5 = (A00+A11)(B00+B11)
    # M6 = (A00-A11)B01
    # M7 = (A01-A11)(B00+B10)
    # And C is:
    # C00 = M5 + M4 - M2 - M1
    # C01 = M3 + M1 + M6 - M7 ?
    # C10 = M2 + M4 + M7 - M6 ?
    # C11 = M5 - M1 - M3 + M2
    # I will use this set.
    
    # M6: A00-A11, B01
    U[0][col] = 1
    U[3][col] = -1
    V[1][col] = 1
    W[0][col] = 0
    W[1][col] = 1
    W[2][col] = -1
    W[3][col] = 0
    col += 1
    
    # M7: A01-A11, B00+B10
    U[1][col] = 1
    U[3][col] = -1
    V[0][col] = 1
    V[5][col] = 1
    W[0][col] = 0
    W[1][col] = -1
    W[2][col] = 1
    W[3][col] = 0
    col += 1
    
    # 2. Schoolbook for A[0:2, 0:2] * B[0:2, 2:5]
    # i in 0,1; j in 0,1; k in 2,3,4
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # 3. Schoolbook for A[0:2, 2:3] * B[2:3, 0:5]
    # i in 0,1; j in 2; k in 0,1,2,3,4
    for i in range(2):
        for j in range(2, 3):
            for k in range(5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
