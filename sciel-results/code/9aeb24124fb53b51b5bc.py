import numpy as np

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 19 construction:
    # Strassen for first 2 columns (rank 7)
    # Schoolbook for last 3 columns (rank 12)
    
    rank_strassen = 7
    rank_schoolbook = 2 * 2 * 3  # 12
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen part for k=0, 1
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Column 0: M1
    # U: A00+A11 -> U[0,0]=1, U[3,0]=1 (i=0,j=0 and i=1,j=1)
    # V: B00+B11 -> V[0,0]=1, V[1,0]=1 (j=0,k=0 and j=1,k=1)
    # W: C00, C11 -> W[0,0]=1, W[1,0]=1 (k=0,i=0 and k=1,i=1)
    
    # We need to be careful with the signs and combinations.
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 - M3 + M2 - M6
    
    # Let's define the 7 terms explicitly.
    
    # Term 0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1  # j=0,k=0 and j=1,k=1
    W[0][0] = 1; W[1][0] = 1  # k=0,i=0 and k=1,i=1
    
    # Term 1: M2
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1               # B00
    W[0][1] = 0; W[1][1] = 1  # C10, C11?
    # C10 = M2 + M6 -> W[1,1] (k=0,i=1) = 1
    # C11 = M1 - M3 + M2 - M6 -> W[2,1] (k=1,i=1) = 1
    W[1][1] = 1  # k=0, i=1
    W[2][1] = 1  # k=1, i=1
    
    # Term 2: M3
    U[0][2] = 1               # A00
    V[2][2] = 1; V[3][2] = -1 # B10, -B11 (j=1,k=0 and j=1,k=1)
    W[0][2] = 0; W[1][2] = 1  # C01 = M3 + M7 -> k=1, i=0
    W[1][2] = 1
    W[2][2] = -1 # C11 = ... - M3 ... -> k=1, i=1
    
    # Term 3: M4
    U[3][3] = 1               # A11
    V[1][3] = 1; V[0][3] = -1 # B01, -B00 (j=0,k=1 and j=0,k=0)
    W[0][3] = -1 # C00 = ... - M4 ... -> k=0, i=0
    W[1][3] = 0
    W[2][3] = 0
    
    # Term 4: M5
    U[0][4] = 1; U[1][4] = 1  # A00, A01
    V[3][4] = 1               # B11 (j=1,k=1)
    W[0][4] = 1 # C00 = ... + M5 ...
    W[1][4] = 0
    
    # Term 5: M6
    U[2][5] = 1; U[0][5] = -1 # A10, -A00
    V[0][5] = 1; V[2][5] = 1  # B00, B10
    W[1][5] = 1 # C10 = M2 + M6
    W[2][5] = -1 # C11 = ... - M6 ...
    
    # Term 6: M7
    U[1][6] = 1; U[3][6] = -1 # A01, -A11
    V[3][6] = 1; V[1][6] = 1  # B11, B01
    W[0][6] = -1 # C00 = ... - M7 ...
    W[1][6] = 1  # C01 = M3 + M7
    
    # Schoolbook for k=2,3,4
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
