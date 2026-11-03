import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Index mapping helpers
    # A is 2x3, indices 0..5
    # B is 3x3, indices 0..8
    # C is 2x3, indices 0..5 (W is p*n = 3*2=6, index k*n+i)
    
    col = 0
    
    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: 0,1,2,3 (A00,A01,A10,A11)
    # B indices: 0,1,2,3 (B00,B01,B10,B11)
    # C indices: 0,1,2,3 (C00,C01,C10,C11) -> W indices: k*n+i
    # C00 (i=0,k=0) -> W[0]
    # C01 (i=0,k=1) -> W[2]
    # C10 (i=1,k=0) -> W[1]
    # C11 (i=1,k=1) -> W[3]
    
    # Strassen terms
    # M1 = (A00+A01)(B00+B01)
    # M2 = (A00+A01)(B10+B11)
    # M3 = (A00+A10)(B00+B10)
    # M4 = (A11-A00)(B00+B01)
    # M5 = (A01+A11)(B10+B11)
    # M6 = (A00+A01-A10-A11)(B10+B11)
    # M7 = (A10-A00+A11-A01)(B00+B01)
    
    # Contributions to C:
    # C00 = M1 - M3 - M4 + M7
    # C01 = M2 - M5 - M4 + M7
    # C10 = M3 - M1 + M6 + M7
    # C11 = M5 - M2 + M6 + M7
    
    # We add 7 columns for these.
    # Col 0: M1
    U[0][col] = 1; U[1][col] = 1
    V[0][col] = 1; V[1][col] = 1
    W[0][col] = 1; W[2][col] = -1; W[1][col] = -1; W[3][col] = 1
    col += 1
    
    # Col 1: M2
    U[0][col] = 1; U[1][col] = 1
    V[2][col] = 1; V[3][col] = 1
    W[2][col] = 1; W[3][col] = -1
    col += 1
    
    # Col 2: M3
    U[0][col] = 1; U[2][col] = 1
    V[0][col] = 1; V[2][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # Col 3: M4
    U[0][col] = -1; U[3][col] = 1
    V[0][col] = 1; V[1][col] = 1
    W[0][col] = -1; W[2][col] = -1
    col += 1
    
    # Col 4: M5
    U[1][col] = 1; U[3][col] = 1
    V[2][col] = 1; V[3][col] = 1
    W[2][col] = -1; W[3][col] = 1
    col += 1
    
    # Col 5: M6
    U[0][col] = 1; U[1][col] = 1; U[2][col] = -1; U[3][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # Col 6: M7
    U[0][col] = -1; U[1][col] = -1; U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[1][col] = 1
    W[0][col] = 1; W[2][col] = 1; W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # 2. Schoolbook for A[0:2, 0:2] * B[0:2, 2:3]
    # A indices: 0,1,2,3
    # B indices: 0,1 (global B indices 6,7? No. B is 3x3. B[0:2, 2:3] is rows 0,1 col 2. Indices 2, 5)
    # C indices: i=0,1; k=2. W indices: 2*n+0=4, 2*n+1=5.
    
    # Term: A00 * B[0,2] (V idx 2) -> W[4]
    U[0][col] = 1; V[2][col] = 1; W[4][col] = 1
    col += 1
    
    # Term: A01 * B[0,2] (V idx 2) -> W[4]
    U[1][col] = 1; V[2][col] = 1; W[4][col] = 1
    col += 1
    
    # Term: A10 * B[1,2] (V idx 5) -> W[5]
    U[2][col] = 1; V[5][col] = 1; W[5][col] = 1
    col += 1
    
    # Term: A11 * B[1,2] (V idx 5) -> W[5]
    U[3][col] = 1; V[5][col] = 1; W[5][col] = 1
    col += 1
    
    # 3. Schoolbook for A[0:2, 2:3] * B[2:3, 0:2]
    # A indices: 4,5
    # B indices: 6,7 (Row 2, cols 0,1)
    # C indices: i=0,1; k=0,1. W indices: 0,1,2,3.
    
    # Term: A02 (U idx 4) * B[2,0] (V idx 6) -> W[0] (i=0,k=0)
    U[4][col] = 1; V[6][col] = 1; W[0][col] = 1
    col += 1
    
    # Term: A02 (U idx 4) * B[2,1] (V idx 7) -> W[2] (i=0,k=1)
    U[4][col] = 1; V[7][col] = 1; W[2][col] = 1
    col += 1
    
    # Term: A12 (U idx 5) * B[2,0] (V idx 6) -> W[1] (i=1,k=0)
    U[5][col] = 1; V[6][col] = 1; W[1][col] = 1
    col += 1
    
    # Term: A12 (U idx 5) * B[2,1] (V idx 7) -> W[3] (i=1,k=1)
    U[5][col] = 1; V[7][col] = 1; W[3][col] = 1
    col += 1
    
    # 4. Schoolbook for A[0:2, 2:3] * B[2:3, 2:3]
    # A indices: 4,5
    # B indices: 8 (Row 2, col 2)
    # C indices: i=0,1; k=2. W indices: 4,5.
    
    # Term: A02 (U idx 4) * B[2,2] (V idx 8) -> W[4]
    U[4][col] = 1; V[8][col] = 1; W[4][col] = 1
    col += 1
    
    # Term: A12 (U idx 5) * B[2,2] (V idx 8) -> W[5]
    U[5][col] = 1; V[8][col] = 1; W[5][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

