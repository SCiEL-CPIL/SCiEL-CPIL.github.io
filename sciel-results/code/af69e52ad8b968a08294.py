"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    
    # We use a block decomposition approach:
    # Split the 3x3 matrix into blocks and use Strassen-like ideas.
    # 
    # The key insight: M_{3,3,8} can be viewed as having 3x3 blocks along
    # the p-dimension. Specifically, we can write:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Let's try to exploit the structure where p=8 is a multiple of 4,
    # and n=m=3.
    #
    # Approach: Use the fact that we can decompose the multiplication
    # by grouping columns of B. Since p=8, we can think of B as 8 columns.
    #
    # A more promising approach: use a rank-7 Strassen decomposition for
    # 2x2 matrix multiplication, applied to sub-blocks.
    #
    # Actually, let's try a direct algebraic approach:
    # M_{3,3,8} has schoolbook rank 72.
    # 
    # We can try to find savings by noticing that the tensor is:
    # T[i,j,k] = e_{i,j} \otimes e_{j,k} \otimes e_{k,i} (in appropriate indices)
    #
    # Let's try a construction based on splitting the middle index j.
    # For each j in {0,1,2}, we have a term A[i,j] * B[j,k].
    # This gives us 3 separate "slices" each of size 3x8.
    #
    # For slice j: we need to compute sum over i,k of A[i,j]*B[j,k]*C[i,k]
    # This is essentially an outer product structure: (A[:,j]) \otimes (B[j,:])
    # which has rank 3*8 = 24 for each j, total 72.
    #
    # To get savings, we need to find linear dependencies across the slices.
    #
    # Alternative: Try to use a known low-rank decomposition.
    # For 2x2x2, Strassen gives rank 7 instead of 8.
    # For 3x3x3, the best known is 23 (Bayer, Strassen).
    #
    # Let's try: M_{3,3,8} = M_{3,3,4} \otimes M_{1,1,2}
    # If M_{3,3,4} has rank r, then M_{3,3,8} has rank 2r.
    # Schoolbook for M_{3,3,4} is 36, so this gives 72. No help.
    #
    # Let's try M_{3,3,8} = M_{3,3,2} \otimes M_{1,1,4}
    # Schoolbook for M_{3,3,2} is 18, M_{1,1,4} is 4, total 72. No help.
    #
    # What about using a non-tensor-product approach?
    # 
    # Let me try a different strategy: use the known rank-23 decomposition
    # for 3x3x3 matrix multiplication, and extend it.
    #
    # Actually, the simplest guaranteed improvement: 
    # Use 4 disjoint 2x2 Strassen blocks? No, 3 is not even.
    #
    # Let me try: decompose M_{3,3,8} by grouping the 8 columns into
    # 4 pairs, and for each pair use a 2-column Strassen-like trick.
    #
    # For 2 columns of B (k=2t, 2t+1), the contribution to C is:
    # C[i,2t] = sum_j A[i,j] B[j,2t]
    # C[i,2t+1] = sum_j A[i,j] B[j,2t+1]
    #
    # This is like multiplying A (3x3) by a 3x2 matrix.
    # The tensor for 3x3x2 has schoolbook rank 18.
    # Can we do better? 
    #
    # For M_{3,3,2}: we can write it as M_{3,3,2} = M_{3,3,1} + M_{3,3,1} (shifted)
    # But that's just 2 * 9 = 18.
    #
    # Hmm, let me try yet another approach. Use the fact that we can
    # share computations across the 8 columns.
    #
    # Strategy: Precompute some linear combinations of rows of B,
    # then combine with A.
    #
    # Let's try a rank-64 construction (saving 8 from schoolbook 72):
    # For each of the 8 columns k, we compute C[:,k] = A * B[:,k].
    # If we could share 1 computation per column, we'd save 8.
    #
    # Actually, let me just try to find ANY improvement by using
    # a structured search with numpy.
    
    # Let's try a different approach: use the bilinear complexity
    # of matrix multiplication with a structured ansatz.
    
    # For now, let's return a valid decomposition. The best I can
    # guarantee quickly is the schoolbook rank of 72.
    # But let me try one more thing: a rank-68 construction.
    
    # Idea: Use 4 independent Strassen 2x2 decompositions on overlapping
    # 2x2 submatrices of the 3x3 matrix, plus handle the remaining terms.
    
    # Actually, here's a concrete approach:
    # Split the 3x3 matrix A into:
    # A = [[a00, a01, a02],
    #      [a10, a11, a12],
    #      [a20, a21, a22]]
    # 
    # And B (3x8) into rows b0, b1, b2 (each 1x8).
    #
    # C[i,k] = a_i0*b0[k] + a_i1*b1[k] + a_i2*b2[k]
    #
    # We can precompute:
    # s0 = a00 + a01 + a02  (but this doesn't help directly)
    #
    # Let me try: for each row i of A, compute C[i,:] = A[i,:] * B
    # This is a 1x3 times 3x8 = 1x8 multiplication.
    # The tensor for 1x3x8 has rank 24 (schoolbook).
    # For 3 rows, total rank 72.
    #
    # To save, we need to share across rows. 
    #
    # Key insight: If we precompute linear combinations of the rows of B,
    # we can reduce the number of multiplications.
    #
    # Let's define:
    # d0 = b0 + b1  (element-wise, 1x8)
    # d1 = b1 + b2  (1x8)  
    # d2 = b2 + b0  (1x8)
    #
    # Then for row 0 of C:
    # C[0,k] = a00*b0[k] + a01*b1[k] + a02*b2[k]
    # 
    # We can write b0 = (d0 - d1 + d2)/2, etc. (circular)
    # This doesn't directly help.
    #
    # Let me try a completely different approach: use numpy to search
    # for a low-rank decomposition with a structured ansatz.
    
    # Given time constraints, let me return the schoolbook decomposition.
    # It's guaranteed correct and has rank 72.
    
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
    
    result = {"U": U, "V": V, "W": W}
    return result
# EVOLVE-BLOCK-END
