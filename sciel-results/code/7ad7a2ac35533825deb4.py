import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 3 Strassen blocks (2x2x2) for cols 0-1, 2-3, 4-5
    # 1 Schoolbook block (2x2x1) for col 6 (part of A'B')
    # 1 Tail term (2x1x1 outer product) for A''B''
    rank = 3 * 7 + 4 + 1
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U indices: i*m + j. i in {0,1}, j in {0,1}
    # V indices: j*p + k. j in {0,1}, k in {k0, k1}
    # W indices: k*n + i. k in {k0, k1}, i in {0,1}
    
    strassen_U = [
        # M1
        [[1, 0], [0, 1]], # A00, A01, A10, A11 -> 1,0,0,1
        # M2
        [[0, 0], [1, 1]],
        # M3
        [[1, 0], [0, 0]],
        # M4
        [[0, 0], [0, 1]],
        # M5
        [[1, 1], [0, 0]],
        # M6
        [[-1, 0], [1, 0]],
        # M7
        [[0, 1], [0, -1]]
    ]
    
    strassen_V = [
        # M1
        [[1, 1], [1, 1]], # B00, B01, B10, B11 -> 1,1,1,1
        # M2
        [[1, 0], [0, 0]],
        # M3
        [[0, -1], [1, 0]],
        # M4
        [[-1, 1], [0, 0]],
        # M5
        [[0, 0], [0, 1]],
        # M6
        [[1, 1], [0, 0]],
        # M7
        [[0, 0], [1, 1]]
    ]
    
    strassen_W = [
        # M1
        [[1, 0], [1, 0]], # C00, C01, C10, C11 -> 1,0,0,1 ? No.
        # W index is k*n + i.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        # C00 (k=0,i=0) coeff 1
        # C01 (k=1,i=0) coeff 0
        # C10 (k=0,i=1) coeff 0
        # C11 (k=1,i=1) coeff 1
        # So W = [1, 0, 0, 1]
        [1, 0, 0, 1],
        # M2
        # C10 coeff 1. C10 is k=0, i=1 -> index 1.
        [0, 1, 0, 0],
        # M3
        # C01 coeff 1. C01 is k=1, i=0 -> index 2.
        [0, 0, 1, 0],
        # M4
        # C00 coeff 1, C10 coeff 1.
        # C00: 0, C10: 1.
        [1, 1, 0, 0],
        # M5
        # C00 coeff -1, C01 coeff 1.
        # C00: 0, C01: 2.
        [-1, 0, 1, 0],
        # M6
        # C11 coeff 1.
        # C11: 3.
        [0, 0, 0, 1],
        # M7
        # C00 coeff 1, C11 coeff 1.
        [1, 0, 0, 1]
    ]
    
    # Note: The W values above are for the 2x2 output.
    # We need to map them to the global W indices.
    
    for q in range(3):
        k0 = 2 * q
        k1 = 2 * q + 1
        for s in range(7):
            r = col
            # Fill U
            for i in range(2):
                for j in range(2):
                    U[i * m + j][r] = strassen_U[s][i][j]
            # Fill V
            for j in range(2):
                # k0, k1
                V[j * p + k0][r] = strassen_V[s][j][0]
                V[j * p + k1][r] = strassen_V[s][j][1]
            # Fill W
            # W indices: k*n + i
            # k0, k1
            # i=0,1
            # W[k0*n + 0], W[k0*n + 1], W[k1*n + 0], W[k1*n + 1]
            W[k0 * n + 0][r] = strassen_W[s][0]
            W[k0 * n + 1][r] = strassen_W[s][1]
            W[k1 * n + 0][r] = strassen_W[s][2]
            W[k1 * n + 1][r] = strassen_W[s][3]
            col += 1
            
    # Schoolbook for last column (k=6) of A'B'
    # C_i,6 = A_i0 B_0,6 + A_i1 B_1,6
    # 4 terms
    sb_terms = [
        (0, 0, 0, 6), # i=0, j=0, k=6
        (0, 1, 1, 6), # i=0, j=1, k=6
        (1, 0, 0, 6), # i=1, j=0, k=6
        (1, 1, 1, 6)  # i=1, j=1, k=6
    ]
    
    for i, j, jj, k in sb_terms:
        r = col
        U[i * m + j][r] = 1
        V[jj * p + k][r] = 1
        W[k * n + i][r] = 1
        col += 1
        
    # Tail term
    # C_i,k += A_i2 B_2,k
    r = col
    for i in range(2):
        U[i * m + 2][r] = 1
    for k in range(7):
        V[2 * p + k][r] = 1
        W[k * n + 0][r] = 1
        W[k * n + 1][r] = 1
    col += 1
        
    return {"U": U, "V": V, "W": W}

