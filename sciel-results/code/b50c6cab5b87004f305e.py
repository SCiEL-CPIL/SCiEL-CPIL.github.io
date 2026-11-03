import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 65: 49 from Strassen 4x4x4, 16 from schoolbook 4x4x1
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. Strassen 4x4x4 part (first 49 columns)
    # Define 2x2 Strassen factors
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Kronecker product for 4x4
    # U4[i*4+j, c1*7+c2] = U2[i, c1] * U2[j, c2]
    # Indices: i,j in 0..3. c1,c2 in 0..6.
    # U2 is 4x7.
    # U4 is 16x49.
    for i in range(4):
        for j in range(4):
            for c1 in range(7):
                for c2 in range(7):
                    val = U2[i][c1] * U2[j][c2]
                    if val != 0:
                        U[i * 4 + j][c1 * 7 + c2] = val
                        
    # V4 for first 4 cols of B (j=0..3, k=0..3)
    # V index: j*4 + k.
    # V4[j*4+k, c1*7+c2] = V2[j, c1] * V2[k, c2]
    for j in range(4):
        for k in range(4):
            for c1 in range(7):
                for c2 in range(7):
                    val = V2[j][c1] * V2[k][c2]
                    if val != 0:
                        # V row index is j*p + k? No, for the 4x4x4 part, 
                        # the B matrix is 4x4.
                        # In the full problem, B is 4x5.
                        # The first 4 columns of B are used here.
                        # So k in 0..3 corresponds to k in 0..3 of the full B.
                        # V index in full V is j*5 + k.
                        V[j * 5 + k][c1 * 7 + c2] = val
                        
    # W4 for first 4 cols of C (k=0..3, i=0..3)
    # W index: k*4 + i.
    # W4[k*4+i, c1*7+c2] = W2[k, c1] * W2[i, c2]
    for k in range(4):
        for i in range(4):
            for c1 in range(7):
                for c2 in range(7):
                    val = W2[k][c1] * W2[i][c2]
                    if val != 0:
                        # W row index is k*n + i.
                        # k in 0..3 corresponds to first 4 cols of C.
                        W[k * 4 + i][c1 * 7 + c2] = val

    # 2. Schoolbook 4x4x1 part (last 16 columns, indices 49..64)
    # C_col4 = A * B_col4
    # C_i = sum_j A_ij B_j4
    # Term (i, j) corresponds to column 49 + i*4 + j
    for i in range(4):
        for j in range(4):
            c = 49 + i * 4 + j
            U[i * 4 + j][c] = 1
            # V index for B_j4 is j*5 + 4
            V[j * 5 + 4][c] = 1
            # W index for C_4i is 4*4 + i
            W[4 * 4 + i][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
