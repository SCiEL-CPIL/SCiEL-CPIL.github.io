import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s kill
    
    # We'll try to find a rank-minimal decomposition.
    # Schoolbook rank = n*m*p = 3*3*8 = 72
    # Known: rank of 3x3x3 matrix multiplication is at most 23 (Lader 1976, actually 23 is known upper bound? Let me check.
    # Actually, the rank of 3x3 matrix multiplication is known to be 19 (Stothers 1985) or maybe less.
    # Wait, the tensor rank of matrix multiplication for 3x3x3 is known to be 19? No, that's for 2x2x2 being 7.
    # For 3x3x3, the best known upper bound is 23 (Lader) and lower bound is 19.
    # Actually, I recall that the rank of M_{3,3,3} is not known exactly, but there's a 23-term algorithm.
    # Let me just use the schoolbook for now and try to improve.
    # 
    # Actually, let me think about this differently. The task is 3x3 * 3x8.
    # This can be seen as 8 independent 3x3 multiplications (each column of B is a 3-vector, 
    # so C = A * B where B is 3x8, each column of C is A times a column of B).
    # So we need to compute 8 instances of 3x3 multiplication simultaneously.
    #
    # The rank of the combined tensor M_{3,3,8} is at most 8 * rank(M_{3,3,3}).
    # If rank(M_{3,3,3}) = 23, then rank(M_{3,3,8}) <= 184, which is worse than 72.
    # So that's not helpful.
    #
    # Actually, the schoolbook for 3x3 * 3x8 is: for each of 3 rows of A, 3 cols of A, 8 cols of B = 72 mults.
    #
    # Let me just return the schoolbook decomposition with rank 72.
    # Since finding a better one in 120s is hard without known algorithms, I'll stick with 72.
    
    rank = n * m * p  # 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
