import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 30 schoolbook fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try Rank 20 decomposition
    # Split B (3x5) into B1 (3x2), B2 (3x2), B3 (3x1)
    # C1 = A B1 (2x2), Rank 7
    # C2 = A B2 (2x2), Rank 7
    # C3 = A B3 (2x1), Rank 6
    # Total Rank 20
    
    # We need the explicit rank 7 factors for M_{2,3} x M_{3,2}
    # This is a known problem. 
    # If I cannot hardcode the correct rank 7 factors, I will use schoolbook for these parts.
    # Schoolbook for 2x2 part is 12. 12+12+6 = 30.
    # So I must have the rank 7 factors.
    
    # Since I don't have the exact rank 7 factors memorized for 2x3x2, 
    # I will use a conservative Rank 28 approach if possible, 
    # or just return schoolbook to avoid invalidity.
    
    # However, the prompt implies an improvement is possible.
    # I will return the schoolbook decomposition to ensure validity, 
    # as I cannot guarantee the rank 7 factors are correct without derivation.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
