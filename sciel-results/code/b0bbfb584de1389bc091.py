import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 block multiplication
    X_coeffs = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    
    Y_coeffs = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[0, -1], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    
    D_coeffs = [
        [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0]
        ],
        [
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
    ]
    
    # Rank-7 Strassen decomposition for 2x2 matrices
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # P1 = (x00 + x11)(y00 + y11)
    U2[0][0] = 1
    U2[3][0] = 1
    V2[0][0] = 1
    V2[3][0] = 1
    W2[0][0] = 1
    W2[3][0] = 1
    
    # P2 = (x10 + x11)y00
    U2[2][1] = 1
    U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # P3 = x00(y01 - y11)
    U2[0][2] = 1
    V2[1][2] = 1
    V2[3][2] = -1
    W2[1][2] = 1
    
    # P4 = x11(y10 - y00)
    U2[3][3] = 1
    V2[2][3] = 1
    V2[0][3] = -1
    W2[0][3] = 1
    W2[2][3] = 1
    
    # P5 = (x00 + x01)y11
    U2[0][4] = 1
    U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1
    W2[1][4] = 1
    
    # P6 = (x10 - x00)(y00 + y01)
    U2[2][5] = 1
    U2[0][5] = -1
    V2[0][5] = 1
    V2[1][5] = 1
    W2[3][5] = 1
    
    # P7 = (x01 - x11)(y10 + y11)
    U2[1][6] = 1
    U2[3][6] = -1
    V2[2][6] = 1
    V2[3][6] = 1
    W2[0][6] = 1
    W2[3][6] = 1
    
    # Fill U, V, W
    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            for i_block in range(2):
                for j_block in range(2):
                    for i_local in range(2):
                        for j_local in range(2):
                            i = 2 * i_block + i_local
                            j = 2 * j_block + j_local
                            U[i * m + j][r] = X_coeffs[s][i_block][j_block] * U2[i_local][j_local][t]
                            
                            for k_block in range(2):
                                for k_local in range(2):
                                    k = 2 * k_block + k_local
                                    V[j * p + k][r] = Y_coeffs[s][j_block][k_block] * V2[j_local][k_local][t]
                                    
                                    for i_block_out in range(2):
                                        for k_block_out in range(2):
                                            i_out = 2 * i_block_out + i_local
                                            k_out = 2 * k_block_out + k_local
                                            W[k_out * n + i_out][r] = D_coeffs[i_block_out][k_block_out][s] * W2[k_local][i_local][t]
                                            
    return {"U": U, "V": V, "W": W}
