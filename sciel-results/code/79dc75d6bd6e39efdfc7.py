import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    We exploit the rectangular structure by treating this as a 3x6x3 multiplication.
    The known lower bound for R(3,6,3) is less than 54 (schoolbook).
    
    For 3x3x3, the Bini-Capovani-Lotti-Roberto (BCLR) algorithm achieves rank 23.
    For 3x3x6, we can use a tensor decomposition approach.
    
    One approach: use the fact that 3x3x6 can be decomposed using algorithms
    for rectangular matrix multiplication.
    
    Let me construct a valid decomposition. The schoolbook gives rank 54.
    We can do better by using known results for rectangular matrix multiplication.
    
    For m=3, n=3, p=6:
    We can write C = A * B where A is 3x3, B is 3x6, C is 3x6.
    
    Using the approach from Strassen's work on rectangular matrix multiplication,
    we can achieve rank less than 54.
    
    Let me use a constructive approach based on tensor rank decompositions.
    
    Actually, let me use a known result: for 3x3x3, rank is 23 (BCLR).
    For 3x3x6, we can potentially use a combination or a direct rectangular algorithm.
    
    Let me try to construct a valid rank-54 (schoolbook) solution first to ensure correctness,
    then see if I can improve it.
    
    Actually, I'll use a known valid decomposition. The key insight is that we need
    integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    
    Let me use the schoolbook approach as a baseline (rank 54) since I need to ensure
    validity, and this is guaranteed to work.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    rank = n * m * p  # 54 for schoolbook
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x 54
    V = [[0] * rank for _ in range(m * p)]  # 18 x 54
    W = [[0] * rank for _ in range(p * n)]  # 18 x 54
    
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
