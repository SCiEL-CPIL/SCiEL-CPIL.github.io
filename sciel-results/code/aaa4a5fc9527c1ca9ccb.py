import numpy as np
import random
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Strategy: Split M=4 into 2+2.
    # For each half, split P=5 into 3+2.
    # Part 1: 3x2x3 (Schoolbook Rank 12)
    # Part 2: 3x2x2 (Schoolbook Rank 12)
    # Total per half: 24.
    # Total: 48.

    R = 48
    U = [[0] * R for _ in range(nm)]
    V = [[0] * R for _ in range(mp)]
    W = [[0] * R for _ in range(pn)]
    
    col = 0
    # Half 1: j in 0,1 (m_sub=2)
    # Sub-part 1: k in 0,1,2 (p_sub=3)
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U[i * 4 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    # Sub-part 2: k in 3,4 (p_sub=2)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i * 4 + j][col] = 1
                V[j * 5 + (k + 3)][col] = 1
                W[(k + 3) * 3 + i][col] = 1
                col += 1
                
    # Half 2: j in 2,3 (m_sub=2)
    # Sub-part 1: k in 0,1,2 (p_sub=3)
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U[i * 4 + (j + 2)][col] = 1
                V[(j + 2) * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    # Sub-part 2: k in 3,4 (p_sub=2)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i * 4 + (j + 2)][col] = 1
                V[(j + 2) * 5 + (k + 3)][col] = 1
                W[(k + 3) * 3 + i][col] = 1
                col += 1

    # Verify
    if col != R:
        return {"U": U_school, "V": V_school, "W": W_school}
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(R):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
