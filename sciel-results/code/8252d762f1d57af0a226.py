"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using 6 blocks of Strassen (rank 7)
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2x2
    # u, v, w are lists of 4 elements corresponding to indices 00, 01, 10, 11
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_w = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 1]
    ]
    
    col = 0
    # 6 blocks: 2 j-blocks x 3 k-blocks
    for j_block in range(2):
        j_offset = j_block * 2
        for k_block in range(3):
            k_offset = k_block * 2
            for s in range(7):
                # Add strassen term s
                for i in range(2):
                    for j_local in range(2):
                        val = strassen_u[s][i * 2 + j_local]
                        if val != 0:
                            j_global = j_offset + j_local
                            U[i * m + j_global][col] += val
                            
                for j_local in range(2):
                    for k_local in range(2):
                        val = strassen_v[s][j_local * 2 + k_local]
                        if val != 0:
                            j_global = j_offset + j_local
                            k_global = k_offset + k_local
                            V[j_global * p + k_global][col] += val
                            
                for k_local in range(2):
                    for i in range(2):
                        val = strassen_w[s][k_local * 2 + i]
                        if val != 0:
                            k_global = k_offset + k_local
                            W[k_global * n + i][col] += val
                            
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
