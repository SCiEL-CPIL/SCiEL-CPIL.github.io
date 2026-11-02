import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Attempt Rank 46 (2 * 23)
    # Define 23 forms for 3x3x3
    # U33: 23 linear forms in A (9 vars)
    # V33: 23 linear forms in B (9 vars)
    # W33: 23 linear forms in C (9 vars)
    
    # I will use a known rank 23 decomposition.
    # If I cannot guarantee correctness, I return schoolbook.
    # For now, I return schoolbook to ensure validity.
    # A true improvement requires the 23 forms.
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
