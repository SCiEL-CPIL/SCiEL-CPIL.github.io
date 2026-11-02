import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: j=0, Rank 16
    # A is 2x1, B is 1x8
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j in {1, 2}, Rank 28
    # A is 2x2, B is 2x8
    # Split B into 4 blocks of 2x2
    # Use Strassen for each block
    # Strassen coefficients
    # P1: (A00+A11)(B00+B11)
    # P2: (A10+A11)B00
    # P3: A00(B10-B11)
    # P4: A11(B01-B00)
    # P5: (A00+A01)B11
    # P6: (A10-A00)(B00+B01)
    # P7: (A01-A11)(B10+B11)
    
    # Mapping indices for A1 (j=1,2)
    # A1[0,0] is A[0,1], A1[0,1] is A[0,2]
    # A1[1,0] is A[1,1], A1[1,1] is A[1,2]
    
    # Mapping indices for B1 (j=1,2)
    # B1[0,0] is B[1, k], B1[0,1] is B[1, k+1]
    # B1[1,0] is B[2, k], B1[1,1] is B[2, k+1]
    
    # Output C1 (i=0,1)
    # C1[0,0] is C[0, k], C1[0,1] is C[0, k+1]
    # C1[1,0] is C[1, k], C1[1,1] is C[1, k+1]
    
    strassen_U = [
        [1, 0, 1, 0], # P1: A00+A11 -> A[0,1] + A[1,2]
        [0, 0, 1, 1], # P2: A10+A11 -> A[1,1] + A[1,2]
        [1, 0, 0, 0], # P3: A00 -> A[0,1]
        [0, 0, 0, 1], # P4: A11 -> A[1,2]
        [1, 1, 0, 0], # P5: A00+A01 -> A[0,1] + A[0,2]
        [-1, 0, 1, 0],# P6: A10-A00 -> A[1,1] - A[0,1]
        [0, 1, 0, -1] # P7: A01-A11 -> A[0,2] - A[1,2]
    ]
    
    # B coefficients for each P
    # B1 indices: 0->j=1,k=2l, 1->j=1,k=2l+1, 2->j=2,k=2l, 3->j=2,k=2l+1
    strassen_V = [
        [1, 0, 0, 1], # P1: B00+B11
        [1, 0, 0, 0], # P2: B00
        [0, 0, 1, -1],# P3: B10-B11
        [0, 1, -1, 0],# P4: B01-B00
        [0, 0, 0, 1], # P5: B11
        [1, 1, 0, 0], # P6: B00+B01
        [0, 0, 1, 1]  # P7: B10+B11
    ]
    
    # W coefficients for each P
    # Output indices: 0->i=0,k=2l, 1->i=0,k=2l+1, 2->i=1,k=2l, 3->i=1,k=2l+1
    # z00 = P1 + P4 - P5 + P7
    # z01 = P3 + P5
    # z10 = P2 + P4
    # z11 = P1 - P2 + P3 + P6
    strassen_W = [
        [1, 0, 1, -1], # P1
        [0, 0, 1, -1], # P2
        [0, 1, 0, 1],  # P3
        [1, 0, 1, 0],  # P4
        [-1, 1, 0, 0], # P5
        [0, 0, 0, 1],  # P6
        [1, 0, 0, 0]   # P7
    ]
    
    for l in range(4):
        k_start = 2 * l
        for s in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    val = strassen_U[s][i * 2 + j]
                    if val != 0:
                        U[i * m + (1 + j)][col] = val
            # V
            for j in range(2):
                for k_idx in range(2):
                    val = strassen_V[s][j * 2 + k_idx]
                    if val != 0:
                        V[(1 + j) * p + (k_start + k_idx)][col] = val
            # W
            for i in range(2):
                for k_idx in range(2):
                    val = strassen_W[s][i * 2 + k_idx]
                    if val != 0:
                        W[(k_start + k_idx) * n + i][col] = val
            col += 1
            
    return {"U": U, "V": V, "W": W}
