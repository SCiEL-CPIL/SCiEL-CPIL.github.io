import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Row 0 of A (1x4x4)
    # A[0, j] * B[j, k] -> C[0, k]
    for j in range(4):
        for k in range(4):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: Rows 1,2 of A (2x4x4)
    # Split into 4 2x2x2 blocks
    # Strassen coefficients for 2x2
    # u, v, w are 2x2 matrices
    # M1
    u1 = [[1, 0], [0, 1]]
    v1 = [[1, 0], [0, 1]]
    w1 = [[1, 0], [0, 1]]
    # M2
    u2 = [[0, 0], [1, 1]]
    v2 = [[1, 0], [0, 0]]
    w2 = [[0, 0], [1, -1]]
    # M3
    u3 = [[1, 0], [0, 0]]
    v3 = [[0, 0], [1, -1]]
    w3 = [[0, 1], [0, 1]]
    # M4
    u4 = [[0, 0], [0, 1]]
    v4 = [[-1, 1], [0, 0]]
    w4 = [[1, 0], [1, 0]]
    # M5
    u5 = [[1, 1], [0, 0]]
    v5 = [[0, 0], [0, 1]]
    w5 = [[-1, 1], [0, 0]]
    # M6
    u6 = [[-1, 0], [1, 0]]
    v6 = [[1, 1], [0, 0]]
    w6 = [[0, 0], [0, 1]]
    # M7
    u7 = [[0, 1], [-1, 0]]
    v7 = [[0, 0], [1, 1]]
    w7 = [[1, 0], [0, 0]]
    
    strassen_terms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # Block 1: A_L (i=1,2; j=0,1) * B_LL (j=0,1; k=0,1)
    i_off = 1
    j_off_A = 0
    j_off_B = 0
    k_off = 0
    for u, v, w in strassen_terms:
        for di in range(2):
            for dj in range(2):
                if u[di][dj] != 0:
                    U[(i_off + di) * m + (j_off_A + dj)][col] += u[di][dj]
        for dj in range(2):
            for dk in range(2):
                if v[dj][dk] != 0:
                    V[(j_off_B + dj) * p + (k_off + dk)][col] += v[dj][dk]
        for di in range(2):
            for dk in range(2):
                if w[di][dk] != 0:
                    W[(k_off + dk) * n + (i_off + di)][col] += w[di][dk]
        col += 1
        
    # Block 2: A_R (i=1,2; j=2,3) * B_RL (j=2,3; k=0,1)
    i_off = 1
    j_off_A = 2
    j_off_B = 2
    k_off = 0
    for u, v, w in strassen_terms:
        for di in range(2):
            for dj in range(2):
                if u[di][dj] != 0:
                    U[(i_off + di) * m + (j_off_A + dj)][col] += u[di][dj]
        for dj in range(2):
            for dk in range(2):
                if v[dj][dk] != 0:
                    V[(j_off_B + dj) * p + (k_off + dk)][col] += v[dj][dk]
        for di in range(2):
            for dk in range(2):
                if w[di][dk] != 0:
                    W[(k_off + dk) * n + (i_off + di)][col] += w[di][dk]
        col += 1
        
    # Block 3: A_L (i=1,2; j=0,1) * B_LR (j=0,1; k=2,3)
    i_off = 1
    j_off_A = 0
    j_off_B = 0
    k_off = 2
    for u, v, w in strassen_terms:
        for di in range(2):
            for dj in range(2):
                if u[di][dj] != 0:
                    U[(i_off + di) * m + (j_off_A + dj)][col] += u[di][dj]
        for dj in range(2):
            for dk in range(2):
                if v[dj][dk] != 0:
                    V[(j_off_B + dj) * p + (k_off + dk)][col] += v[dj][dk]
        for di in range(2):
            for dk in range(2):
                if w[di][dk] != 0:
                    W[(k_off + dk) * n + (i_off + di)][col] += w[di][dk]
        col += 1
        
    # Block 4: A_R (i=1,2; j=2,3) * B_RR (j=2,3; k=2,3)
    i_off = 1
    j_off_A = 2
    j_off_B = 2
    k_off = 2
    for u, v, w in strassen_terms:
        for di in range(2):
            for dj in range(2):
                if u[di][dj] != 0:
                    U[(i_off + di) * m + (j_off_A + dj)][col] += u[di][dj]
        for dj in range(2):
            for dk in range(2):
                if v[dj][dk] != 0:
                    V[(j_off_B + dj) * p + (k_off + dk)][col] += v[dj][dk]
        for di in range(2):
            for dk in range(2):
                if w[di][dk] != 0:
                    W[(k_off + dk) * n + (i_off + di)][col] += w[di][dk]
        col += 1
        
    return {"U": U, "V": V, "W": W}
