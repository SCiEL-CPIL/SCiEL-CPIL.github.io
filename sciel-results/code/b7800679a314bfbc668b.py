import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition using Strassen for 2x2 blocks and Schoolbook for 3x3 blocks
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # a_s[x*2+z], b_s[z*2+y], d_s[y*2+x]
    # Indices for 2x2 matrix: 0:00, 1:01, 2:10, 3:11
    
    # M1
    a0 = [1, 0, 0, 1]
    b0 = [1, 0, 0, 1]
    d0 = [1, 0, 0, 1]
    # M2
    a1 = [0, 0, 1, 1]
    b1 = [1, 0, 0, 0]
    d1 = [0, 0, 1, -1]
    # M3
    a2 = [1, 0, 0, 0]
    b2 = [0, 1, 0, -1]
    d2 = [0, 1, 0, 1]
    # M4
    a3 = [0, 0, 0, 1]
    b3 = [0, 0, 1, -1]
    d3 = [1, 0, 1, 0]
    # M5
    a4 = [1, 1, 0, 0]
    b4 = [0, 0, 0, 1]
    d4 = [-1, 1, 0, 0]
    # M6
    a5 = [0, 0, 1, -1]
    b5 = [1, 1, 0, 0]
    d5 = [0, 0, 0, 1]
    # M7
    a6 = [0, 1, 0, -1]
    b6 = [0, 0, 1, 1]
    d6 = [1, 0, 0, 0]
    
    strassen_a = [a0, a1, a2, a3, a4, a5, a6]
    strassen_b = [b0, b1, b2, b3, b4, b5, b6]
    strassen_d = [d0, d1, d2, d3, d4, d5, d6]
    
    col = 0
    for u in range(3):
        for v in range(3):
            for w in range(3):
                for s in range(7):
                    a_s = strassen_a[s]
                    b_s = strassen_b[s]
                    d_s = strassen_d[s]
                    
                    # Update U
                    for x in range(2):
                        for z in range(2):
                            i = 2 * u + x
                            j = 2 * v + z
                            U[i * m + j][col] = a_s[x * 2 + z]
                            
                    # Update V
                    for z in range(2):
                        for y in range(2):
                            j = 2 * v + z
                            k = 2 * w + y
                            V[j * p + k][col] = b_s[z * 2 + y]
                            
                    # Update W
                    for y in range(2):
                        for x in range(2):
                            k = 2 * w + y
                            i = 2 * u + x
                            W[k * n + i][col] = d_s[y * 2 + x]
                            
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
