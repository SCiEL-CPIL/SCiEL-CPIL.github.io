import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    
    # We'll build a decomposition using the Winograd F(2) scheme for the 2x2 blocks.
    # The idea: split the 4 columns into two 2-column blocks, and for each block,
    # use the rank-4 Winograd scheme for 2x2 matrix multiplication.
    # 
    # For a 2x2 matrix multiplication C = A * B where A is 2x2 and B is 2x2:
    # We can write this as a cyclic convolution over Z_5 or use the explicit rank-4 form.
    #
    # The rank-4 Winograd F(2) decomposition for 2x2 matrix multiplication:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A * B
    #
    # The 4 multiplications are:
    # m0 = (a00 + a10) * (b00 + b10)
    # m1 = (a00 - a10) * (b00 - b10)
    # m2 = (a00 + a01) * (b01 + b11)
    # m3 = (a00 - a01) * (b01 - b11)
    #
    # Then:
    # C00 = m0 + m1 + m2 - m3  (wait, this doesn't look right)
    #
    # Let me use the correct Winograd F(2) scheme:
    # For 2x2 matrix multiplication, the rank is 4 (not 7 as in Strassen).
    # Actually, the rank of the 2x2 matrix multiplication tensor is 7 (Strassen).
    # The Winograd F(2) scheme gives a different decomposition.
    #
    # Let me reconsider. The task mentions "rank-4 decomposition of 2x2 matrix multiplication 
    # via the (2,2) Winograd scheme". This is unusual because the standard rank is 7.
    # However, if we're doing a specific type of convolution or if there's additional structure,
    # we might be able to use a lower rank.
    #
    # Actually, I think the key insight is that we're not doing general 2x2 matrix multiplication,
    # but rather exploiting the structure of the 3x4x6 problem. Let me think about this differently.
    #
    # Alternative approach: Use the known bound rank(M_{3,4,6}) <= 56.
    # One way to achieve this: split the 4 middle dimension into 2+2, and for each 2-column block,
    # we have a 3x2x6 problem. The rank of M_{3,2,6} is at most 18 (schoolbook).
    # So total would be 2 * 18 = 36. But that's too optimistic.
    #
    # Let me use a more careful construction. I'll implement the schoolbook first to ensure
    # correctness, then try to optimize.
    
    rank = n * m * p
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
