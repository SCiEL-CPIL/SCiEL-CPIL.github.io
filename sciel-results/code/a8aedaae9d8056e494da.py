import numpy as np
import itertools

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
    
    # Known rank 23 decomposition for 3x3x3 matrix multiplication
    # This is based on the work of Strassen and subsequent improvements.
    # We define the 23 linear forms for A, B, and C.
    
    # Forms for A (9 variables: a00..a22)
    # Forms for B (9 variables: b00..b22)
    # Forms for C (9 variables: c00..c22)
    
    # We use a specific known rank 23 decomposition.
    # The coefficients are integers or half-integers.
    
    # Let's use the decomposition from "A 23-multiplication algorithm for 3x3 matrix multiplication"
    # The forms are:
    
    # A forms (u):
    u_forms = [
        [1, 0, 0, 0, 1, 0, 0, 0, 1], # a00 + a11 + a22
        [0, 1, 0, 1, 0, 0, 0, 0, 0], # a01 + a10
        [0, 0, 1, 0, 0, 1, 0, 0, 0], # a02 + a21
        [1, 0, 0, 0, -1, 0, 0, 0, 0], # a00 - a11
        [0, 1, 0, -1, 0, 0, 0, 0, 0], # a01 - a10
        [0, 0, 1, 0, 0, -1, 0, 0, 0], # a02 - a21
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # a00
        [0, 1, 0, 0, 0, 0, 0, 0, 0], # a01
        [0, 0, 1, 0, 0, 0, 0, 0, 0], # a02
        [0, 0, 0, 1, 0, 0, 0, 0, 0], # a10
        [0, 0, 0, 0, 1, 0, 0, 0, 0], # a11
        [0, 0, 0, 0, 0, 1, 0, 0, 0], # a12
        [0, 0, 0, 0, 0, 0, 1, 0, 0], # a20
        [0, 0, 0, 0, 0, 0, 0, 1, 0], # a21
        [0, 0, 0, 0, 0, 0, 0, 0, 1], # a22
        [1, 1, 0, 0, 0, 0, 0, 0, 0], # a00 + a01
        [0, 0, 1, 1, 0, 0, 0, 0, 0], # a02 + a10
        [0, 0, 0, 0, 1, 1, 0, 0, 0], # a11 + a12
        [0, 0, 0, 0, 0, 0, 1, 1, 0], # a20 + a21
        [0, 0, 0, 0, 0, 0, 0, 0, 1], # a22
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # a00
        [0, 1, 0, 0, 0, 0, 0, 0, 0], # a01
        [0, 0, 0, 1, 0, 0, 0, 0, 0]  # a10
    ]
    
    # This approach of hardcoding 23 forms is error-prone without a verified source.
    # Instead, let's try to construct a rank 46 decomposition by taking two rank 23 decompositions
    # for the two 3x3 blocks of the 3x6 matrix.
    
    # However, since I cannot guarantee the correctness of the rank 23 forms without a lookup table,
    # I will return the schoolbook decomposition to ensure validity.
    # A future improvement could involve embedding a verified rank 23 table.
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
