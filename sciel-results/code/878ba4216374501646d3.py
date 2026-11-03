import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a decomposition of rank 81.
    # 49 terms for 4x4x4 (Strassen recursive)
    # 32 terms for 4x4x2 (Schoolbook)
    rank = 81
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 factors
    # Indices: 0:00, 1:01, 2:10, 3:11
    U_2x2 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    V_2x2 = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W_2x2 = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 0],
        [-1, 0, 0, 1]
    ]
    
    # Block Strassen coefficients for 4x4
    # r=0..6
    # u[r][I][J], v[r][J][K], w[r][I][K]
    u_block = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[1, 0], [1, 0]],
        [[0, 1], [0, 1]]
    ]
    v_block = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [1, -1]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    w_block = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 0]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 0]],
        [[-1, 0], [0, 1]]
    ]
    
    col = 0
    for r in range(7):
        for s in range(7):
            # This term corresponds to block r, scalar s
            for I in range(2):
                for J in range(2):
                    if u_block[r][I][J] != 0:
                        for i_prime in range(2):
                            for j_prime in range(2):
                                if U_2x2[s][i_prime * 2 + j_prime] != 0:
                                    i = 2 * I + i_prime
                                    j = 2 * J + j_prime
                                    U[i * m + j][col] += U_2x2[s][i_prime * 2 + j_prime] * u_block[r][I][J]
            
            for J in range(2):
                for K in range(2):
                    if v_block[r][J][K] != 0:
                        for j_prime in range(2):
                            for k_prime in range(2):
                                if V_2x2[s][j_prime * 2 + k_prime] != 0:
                                    j = 2 * J + j_prime
                                    k = 2 * K + k_prime
                                    V[j * p + k][col] += V_2x2[s][j_prime * 2 + k_prime] * v_block[r][J][K]
            
            for K in range(2):
                for I in range(2):
                    if w_block[r][I][K] != 0:
                        for k_prime in range(2):
                            for i_prime in range(2):
                                if W_2x2[s][k_prime * 2 + i_prime] != 0:
                                    k = 2 * K + k_prime
                                    i = 2 * I + i_prime
                                    W[k * n + i][col] += W_2x2[s][k_prime * 2 + i_prime] * w_block[r][I][K]
            col += 1
    
    # Schoolbook for 4x4x2 (k=4,5)
    for i in range(n):
        for j in range(m):
            for k in range(4, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
