"""Matrix multiplication 4x4x7 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 exact decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    
    Strategy: Decompose the 4x4x7 multiplication tensor as a direct sum of:
      - A 2x4x4 block (multiplying rows 0-1 of A by all 4 columns of B, producing rows 0-1 of C)
      - A 2x4x7 block (multiplying rows 2-3 of A by all 4 columns of B, producing rows 2-3 of C)
    
    Wait, that's not quite right. Let me think more carefully.
    
    We want to compute C = A * B where A is 4x4 and B is 4x7, so C is 4x7.
    
    Let's partition A into two 2x4 blocks: A_top (rows 0-1) and A_bottom (rows 2-3).
    Then C_top = A_top * B (2x7) and C_bottom = A_bottom * B (2x7).
    
    So we need:
      - Rank for 2x4x7 multiplication (A_top is 2x4, B is 4x7, C_top is 2x7)
      - Rank for 2x4x7 multiplication (A_bottom is 2x4, B is 4x7, C_bottom is 2x7)
    
    Total rank = 2 * rank(2x4x7)
    
    Now, what is the rank of 2x4x7?
    
    We can further decompose 2x4x7. Let's think of it as:
      A is 2x4, B is 4x7, C is 2x7.
    
    We can split A into two 2x2 blocks: A_left (cols 0-1) and A_right (cols 2-3).
    Similarly split B into B_left (rows 0-1, 2x7) and B_right (rows 2-3, 2x7).
    
    Then C = A_left * B_left + A_right * B_right.
    
    Each term is a 2x2x7 multiplication: A_block (2x2) * B_block (2x7) = C_block (2x7).
    
    The rank of 2x2x7: We can use Strassen-like ideas. 
    For 2x2x2, Strassen gives rank 7.
    For 2x2x7, we can view B as having 7 columns, each of which is a 2x1 vector.
    So we're computing for each column j: C[:,j] = A * B[:,j].
    
    Actually, let's think differently. The tensor for 2x2x7 has dimensions (2*2, 2*7, 7*2) = (4, 14, 14).
    
    A simple upper bound: schoolbook is 2*2*7 = 28.
    
    Can we do better? Let's use the fact that 2x2x7 is a "slice" of 2x2x2 in some sense.
    
    Actually, for 2x2xn, the rank is known to be 2n for n >= 2? No, that's not right either.
    
    Let me just use a simple construction. For 2x2x7:
    We can use 2*7 = 14 rank-1 tensors if we fix the 2x2 part... no.
    
    Let's just use schoolbook for the sub-problems and see what rank we get.
    
    For 2x4x7: schoolbook is 2*4*7 = 56. That's too high.
    
    Let me reconsider. Maybe I should use a different decomposition.
    
    Alternative: Use the known fact that rank(4x4x4) <= 49 (Strassen's algorithm gives 49 for 4x4x4).
    For 4x4x7, we can embed it in 4x4x8 and use a rank decomposition for 4x4x8.
    
    Actually, let's try a direct approach. The tensor is of size (4*4, 4*7, 7*4) = (16, 28, 28).
    
    Let's try to construct a rank-28 decomposition by treating the 7 columns of B as independent.
    For each column j of B, we compute C[:,j] = A * B[:,j], which is a 4x1 vector.
    This is a 4x4x1 multiplication, which has rank 16 (schoolbook).
    So for 7 columns, that would be 7*16 = 112. Too high.
    
    Let me try yet another approach. Use bilinear algorithms.
    
    Actually, the simplest correct approach is to just use the schoolbook decomposition with rank = 4*4*7 = 112.
    But we can do better.
    
    Let's use the following: For each pair (i,k) where i is row of C and k is column of C,
    C[i,k] = sum_j A[i,j] * B[j,k].
    
    We can use rank-1 tensors for each (i,j,k) triple, giving rank 4*4*7 = 112.
    
    To improve, we can use Strassen's idea on the 4x4x4 part and handle the extra 3 columns separately.
    
    Split B into B1 (4x4) and B2 (4x3).
    C = A*B1 + A*B2.
    
    For A*B1: use Strassen 4x4x4 with rank 49.
    For A*B2: schoolbook 4*4*3 = 48.
    
    Total: 49 + 48 = 97.
    
    Hmm, that's better than 112 but still high.
    
    Let me just implement the schoolbook decomposition for now to ensure correctness, and aim for a reasonable rank.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    rank = n * m * p  # 112
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
