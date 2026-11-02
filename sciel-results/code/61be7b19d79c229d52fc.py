import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    num_u = n * m  # 9
    num_v = m * p  # 21
    num_w = p * n  # 21
    
    # We aim for Rank 55: 23 (for 3x3x3) + 23 (for 3x3x3) + 9 (for 3x3x1)
    rank = 55
    
    U = np.zeros((num_u, rank))
    V = np.zeros((num_v, rank))
    W = np.zeros((num_w, rank))
    
    # --- Rank-23 Factors for 3x3x3 ---
    # These are the Winograd rank-23 decomposition factors.
    # U, V, W are 9x23 matrices.
    
    # I will use a known set of rank-23 factors.
    # If I cannot recall them exactly, I will fall back to rank-27 schoolbook for the 3x3 parts.
    # However, I will attempt to use the rank-23 factors.
    
    # For safety and correctness, I will use the Schoolbook (Rank 27) for the 3x3 parts 
    # and Rank 9 for the 3x1 part, resulting in Rank 63, 
    # UNLESS I can verify the Rank-23 factors.
    
    # Since I cannot verify the Rank-23 factors from memory with 100% confidence, 
    # and an invalid solution scores 0, I will use a **Rank-57** solution 
    # based on Strassen (Rank 24) for the 3x3 parts.
    
    # Strassen Rank-24 for 3x3x3:
    # Partition 3x3 into 2x2, 2x1, 1x2, 1x1.
    # This is complex.
    
    # Let's stick to the **Rank-63** Schoolbook solution to ensure validity, 
    # as the current program does.
    
    # However, I can improve it by using **Rank-54** if I find a way.
    
    # Let's try to implement the **Rank-23** factors from a standard reference.
    # The factors are integers.
    
    # U factors (9x23)
    U_23 = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    # This is just a placeholder. I don't have the real Rank-23 factors.
    
    # I will return the Schoolbook solution (Rank 63) to ensure validity.
    
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
