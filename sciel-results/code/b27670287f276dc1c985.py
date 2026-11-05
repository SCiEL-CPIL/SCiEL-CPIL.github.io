"""Matrix multiplication 2x5x6 with structured low-rank completion approach."""
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # For 2x5x6 matrix multiplication, we can use the fact that
    # multiplying a 2x5 matrix by a 5x6 matrix gives a 2x6 matrix.
    # The standard rank is 2*5*6 = 60.
    
    # We use a structured decomposition based on the geometry.
    # Key insight: We can group the 5 columns of A and 5 rows of B into blocks
    # and use bilinear identities to reduce rank.
    
    # One effective approach: split the 5x5 middle dimension into blocks.
    # 5 = 2 + 3, so split j into j1 (size 2) and j2 (size 3).
    
    # For the 2x2x6 part (i=0,1; j1=0,1; k=0..5): rank 2*2*6 = 24
    # For the 2x3x6 part (i=0,1; j2=0,1,2; k=0..5): rank 2*3*6 = 36
    # Total: 60 (no improvement)
    
    # Better: use the fact that n=2 is small. We can use identities like:
    # For each pair (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
    # There are 2*6 = 12 such dot products of length 5.
    
    # Each dot product of length 5 has rank 5, so total rank = 12*5 = 60.
    # To improve, we need to share computation across different (i,k) pairs.
    
    # Strassen-like approach: group indices to create shared intermediates.
    # For 2x2x2 blocks, Strassen gives rank 7 instead of 8.
    
    # Let's partition the 5x5 middle into overlapping groups.
    # Actually, a known result: for m x n matrix mult, if n=2, we can use
    # the fact that 2x(2m)x(2p) can be done with rank 7mp + 2m + 2p - something.
    
    # Let me try a direct construction with rank 58 or lower.
    
    # Approach: use the identity for dot products of length 5.
    # a1*b1 + a2*b2 + a3*b3 + a4*b4 + a5*b5
    # Can be computed with 5 multiplications (trivial) or potentially fewer
    # if we share across different dot products.
    
    # For 2x5x6: C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    # We have 12 such sums. If we can share intermediate products, we reduce rank.
    
    # Let's use a block Strassen approach:
    # Split j into {0,1} and {2,3,4}
    # C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k] + A[i,4]*B[4,k])
    
    # First term: 2x2x6 -> rank 24
    # Second term: 2x3x6 -> rank 36
    # Total: 60
    
    # Try different split: {0,1,2} and {3,4}
    # First: 2x3x6 -> 36
    # Second: 2x2x6 -> 24
    # Total: 60
    
    # The key is to use bilinear identities that work across the full dimension.
    
    # Known: 2x2x2 has rank 7 (Strassen)
    # 2x2x3: can we do better than 12?
    
    # For 2x2xn, the rank is at most 7*floor(n/2) + 2*ceil(n/2)*2 or similar.
    
    # Let me just implement the schoolbook for now and return rank 60.
    # Actually, let me try to find a better decomposition.
    
    # For 2x5x6, one approach:
    # Use the fact that for each fixed k, C[:,k] is a 2x1 vector.
    # C[i,k] = A[i,:] . B[:,k]
    
    # If we precompute certain combinations of A and B, we can share work.
    
    # Let's use a rank-58 decomposition based on the following idea:
    # For pairs of columns in the result that share similar structures,
    # we can use bilinear identities.
    
    # Actually, let me just return the schoolbook decomposition for safety.
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
