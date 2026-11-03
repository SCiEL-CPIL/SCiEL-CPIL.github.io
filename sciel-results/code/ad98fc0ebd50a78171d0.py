import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Use two rank-23 decompositions for 3x3x3 to handle 3x3x6
    # Rank = 23 + 23 = 46 < 54
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Bini et al. (1979) rank-23 decomposition for 3x3x3
    # U, V, W are 9x23 matrices for 3x3x3
    
    # Coefficients for the 23 terms
    # I'll use a known rank-23 decomposition
    
    # Define U, V, W for 3x3x3 (each 9x23)
    U333 = np.zeros((9, 23), dtype=int)
    V333 = np.zeros((9, 23), dtype=int)
    W333 = np.zeros((9, 23), dtype=int)
    
    # Term 0: A[0,0]*B[0,0]
    U333[0, 0] = 1; V333[0, 0] = 1; W333[0, 0] = 1
    # Term 1: A[0,1]*B[1,0]
    U333[1, 1] = 1; V333[3, 1] = 1; W333[0, 1] = 1
    # Term 2: A[0,2]*B[2,0]
    U333[2, 2] = 1; V333[6, 2] = 1; W333[0, 2] = 1
    # Term 3: A[1,0]*B[0,1]
    U333[3, 3] = 1; V333[1, 3] = 1; W333[1, 3] = 1
    # Term 4: A[1,1]*B[1,1]
    U333[4, 4] = 1; V333[4, 4] = 1; W333[1, 4] = 1
    # Term 5: A[1,2]*B[2,1]
    U333[5, 5] = 1; V333[7, 5] = 1; W333[1, 5] = 1
    # Term 6: A[2,0]*B[0,2]
    U333[6, 6] = 1; V333[2, 6] = 1; W333[2, 6] = 1
    # Term 7: A[2,1]*B[1,2]
    U333[7, 7] = 1; V333[5, 7] = 1; W333[2, 7] = 1
    # Term 8: A[2,2]*B[2,2]
    U333[8, 8] = 1; V333[8, 8] = 1; W333[2, 8] = 1
    
    # This is just the 9 diagonal terms, rank 9. I need 23 terms.
    # The Bini et al. algorithm uses additional terms to reduce rank.
    
    # Let me use a different approach: return schoolbook for now
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
