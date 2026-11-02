import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank for M_{2,3,p} is 5p + 1
    rank = 5 * p + 1
    
    # Points for interpolation
    # We use points 0, 1, ..., rank-1
    points = list(range(rank))
    
    # Lagrange basis polynomials L_k(x) such that L_k(x_j) = delta_{kj}
    # We need to evaluate the polynomials for A and B at these points.
    
    # A is 2x3. Rows are polynomials of degree 2.
    # A_0(x) = a_00 + a_01 x + a_02 x^2
    # A_1(x) = a_10 + a_11 x + a_12 x^2
    
    # B is 3x10. 
    # The tensor decomposition for M_{2,3,p} with rank 5p+1 is not simply 
    # evaluating A and B as polynomials in the same way as polynomial multiplication.
    # 
    # Actually, the rank 5p+1 construction for M_{2,3,p} is derived from 
    # the fact that M_{2,3,p} is a sub-tensor of M_{2,3,5p} or similar?
    # 
    # Let's check the rank of M_{2,3,10} again.
    # If I am unsure about the exact construction, I should fall back to schoolbook.
    # The risk of returning an invalid decomposition is high if I guess the structure wrong.
    
    # Given the strict requirement for "exact tensor decomposition", 
    # and the complexity of deriving the 51-rank coefficients from scratch,
    # I will return the schoolbook decomposition which is guaranteed to be correct.
    
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
