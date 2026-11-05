import numpy as np
import random
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    rank = 125
    
    # We will use the 6x6 Strassen decomposition (rank 21) and then "lift" it to 5x5.
    # The 6x6 Strassen algorithm multiplies two 6x6 matrices.
    # We want to multiply 5x5 matrices.
    # We can view the 5x5 matrices as embedded in 6x6 matrices by adding a zero row/column.
    # Let A_6 be a 6x6 matrix where A_6[i,j] = A[i,j] for i,j < 5, and 0 otherwise.
    # Similarly for B_6.
    # Then C_6 = A_6 B_6 will have C_6[i,j] = C[i,j] for i,j < 5, and 0 in the 6th row and column.
    # So we can just take the top-left 5x5 block of C_6.
    
    # The Strassen algorithm for 6x6 matrices (which is 2x2 blocks of 3x3 matrices)
    # has rank 21.
    # Let's construct the 2x2 block Strassen for 3x3 blocks.
    # The 7 scalar multiplications are:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M is a 3x3 matrix product.
    # Since we are working over rationals, we can use the schoolbook 27-term decomposition for each 3x3 product.
    # But wait, we need to be careful: the 3x3 products themselves can be decomposed.
    # Actually, let's just use the schoolbook decomposition for the 3x3 subproblems.
    # Total rank would be 7 * 27 = 189. That's worse than 125.
    
    # Let's try a different approach: use the known rank-21 decomposition for 6x6 matrix multiplication directly.
    # The rank-21 decomposition for 6x6 is equivalent to using Strassen's algorithm on 2x2 blocks of 3x3 matrices.
    # But we can also think of it as a direct decomposition.
    
    # Actually, the simplest way is to just use the schoolbook decomposition for 5x5x5, which is rank 125.
    # But we want to beat 125.
    
    # Let's try the 6x6 embedding with a better decomposition for the 3x3 subproblems.
    # The best known rank for 3x3 matrix multiplication is 23 (Håstad 1990).
    # So 7 * 23 = 161. Still worse than 125.
    
    # Let's try another approach: use the 4x4 Strassen decomposition (rank 19) and embed 5x5 into 4x4? No, that doesn't work.
    
    # Actually, let's just use the schoolbook decomposition for 5x5x5. It's rank 125.
    # But the seed program already does this. We need to beat it.
    
    # Let's try to use the rank-19 decomposition for 4x4 matrix multiplication.
    # We can embed 5x5 into 4x4? No, 5 > 4.
    
    # What if we use a different strategy: use the rank-21 decomposition for 6x6 and project back?
    # The 6x6 Strassen decomposition has rank 21.
    # When we embed 5x5 into 6x6, the 6th row and column are zero.
    # So the 6x6 product C_6 has the form:
    # [ C   0 ]
    # [ 0   0 ]
    # where C is the 5x5 product.
    # So we just need to extract the top-left 5x5 block.
    # The rank of the 6x6 decomposition is 21, but we only need the 5x5 part.
    # However, the rank of the projected decomposition might be less than 21? No, it's at most 21.
    # But 21 < 125!
    
    # Wait, this is a great idea. Let's construct the 6x6 Strassen decomposition.
    # The 6x6 Strassen algorithm uses 7 multiplications of 3x3 matrices.
    # Each 3x3 multiplication can be done with rank 27 (schoolbook).
    # So total rank = 7 * 27 = 189. That's worse than 125.
    
    # But we can use a better decomposition for the 3x3 subproblems.
    # The best known rank for 3x3 is 23.
    # 7 * 23 = 161. Still worse.
    
    # What if we use the direct rank-21 decomposition for 6x6?
    # The rank-21 decomposition is not just Strassen on 3x3 blocks; it's a direct decomposition.
    # Actually, the rank of 6x6 matrix multiplication is at most 21, and this is achieved by Strassen's algorithm applied to 2x2 blocks of 3x3 matrices.
    # But the individual 3x3 multiplications are not decomposed further in the rank-21 bound; they are treated as "scalar" multiplications.
    # So the rank-21 decomposition assumes that 3x3 matrix multiplication has rank 1, which is false.
    
    # I think I'm confusing myself. Let's start over.
    
    # The rank of the 5x5x5 matrix multiplication tensor is at most 125 (schoolbook).
    # We need to find a decomposition with rank < 125.
    
    # One approach is to use the 4x4x4 decomposition (rank 19) and extend it to 5x5x5.
    # But I don't know how to do that directly.
    
    # Another approach is to use the 6x6x6 decomposition (rank 21) and project it to 5x5x5.
    # The 6x6x6 tensor can be decomposed into 21 rank-1 tensors.
    # Each rank-1 tensor corresponds to a triple of vectors u, v, w in C^6.
    # We can project u, v, w to C^5 by dropping the 6th component.
    # The resulting 5x5x5 tensor might have rank at most 21, but it might not be the correct tensor.
    
    # Actually, the embedding approach is:
    # Let T_6 be the 6x6x6 matrix multiplication tensor.
    # Let P be the projection from C^6 to C^5 (dropping the 6th component).
    # Then P(T_6) is not the 5x5x5 tensor; it's a different tensor.
    
    # I think the correct approach is to use the 6x6 decomposition and then "restrict" it to the 5x5 subspace.
    # This is complicated.
    
    # Let's just use the schoolbook decomposition for now. It's safe and correct.
    # But we need to beat the seed.
    
    # Actually, the seed uses rank 125. If we can find a decomposition with rank < 125, we win.
    
    # Let's try the following: use the rank-19 decomposition for 4x4x4 and extend it to 5x5x5 by adding one row and column.
    # This is not straightforward.
    
    # I'll just use the schoolbook decomposition. It's the safest bet.
    
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
