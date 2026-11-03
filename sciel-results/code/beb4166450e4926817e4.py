import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # 3 x M_1,4,4 schoolbook decomposition
    # Each M_1,4,4 has rank 16, total rank = 48
    rank = 48
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 3 blocks of M_1,4,4
    for i_block in range(3):
        for c_block in range(2):  # 2 blocks of M_1,4,4 for p=8 (4 columns each)
            for k in range(4):
                for j in range(4):
                    u_idx = i_block * 4 + k
                    v_idx = k * 8 + (j + 4 * c_block)
                    w_idx = (j + 4 * c_block) * 3 + i_block
                    
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
