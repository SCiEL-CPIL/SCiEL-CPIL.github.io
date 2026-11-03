import numpy as np

def construct():
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: 0,1, 2,3 (i*m+j)
    # B indices: 0,1, 3,4 (j*p+k)
    # C indices: 0,1, 2,3 (k*n+i)
    
    # Strassen 7 terms
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[1][col] = 1; W[2][col] = 1; W[3][col] = -1
    col += 1
    
    # M2 = (A10+A11)B00
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M3 = A00(B10-B00)
    U[0][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    # M4 = A11(B01-B11)
    U[3][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[3][col] = 1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B11+B21) -> Wait, B11 is B[1,1], B21 is B[2,1]?
    # In this sub-problem, B is 2x2 (j=0,1).
    # So B indices are 0,1,3,4.
    # B11 is index 4.
    # There is no B21 in this sub-problem.
    # M7 = (A01-A11)(B11+B01)
    U[1][col] = 1; U[3][col] = -1
    V[4][col] = 1; V[1][col] = 1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # Part 2: Outer product A[0:2, 2] * B[2, 0:2]
    # A indices: 2,3 (j=2)
    # B indices: 6,7 (j=2, k=0,1)
    # C indices: 0,1, 2,3
    
    # T1: A02 * B20
    U[2][col] = 1
    V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # T2: A02 * B21
    U[2][col] = 1
    V[7][col] = 1
    W[1][col] = 1
    col += 1
    
    # T3: A12 * B20
    U[3][col] = 1
    V[6][col] = 1
    W[2][col] = 1
    col += 1
    
    # T4: A12 * B21
    U[3][col] = 1
    V[7][col] = 1
    W[3][col] = 1
    col += 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2]
    # A indices: 0,1,2, 3,4,5
    # B indices: 2, 5, 8 (k=2)
    # C indices: 4, 5 (k=2)
    
    # C02 = A00*B02 + A01*B12 + A02*B22
    U[0][col] = 1; V[2][col] = 1; W[4][col] = 1
    col += 1
    U[1][col] = 1; V[5][col] = 1; W[4][col] = 1
    col += 1
    U[2][col] = 1; V[8][col] = 1; W[4][col] = 1
    col += 1
    
    # C12 = A10*B02 + A11*B12 + A12*B22
    U[3][col] = 1; V[2][col] = 1; W[5][col] = 1
    col += 1
    U[4][col] = 1; V[5][col] = 1; W[5][col] = 1
    col += 1
    U[5][col] = 1; V[8][col] = 1; W[5][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
