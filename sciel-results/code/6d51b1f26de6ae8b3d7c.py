"""Tensor decomposition for 2x4 * 4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Build a rank-reduced decomposition by exploiting structure.
    # For 2x4*4x6, schoolbook is 48. We use a "block" approach:
    # Split the middle dimension into two halves (2+2), giving two 2x2*2x6 blocks.
    # But a better approach: use the fact that rank of matrix mult (n,m,p) is at most
    # n*p*ceil(m/2) when m is even, via a standard reduction.
    #
    # Standard result: M(n,m,p) <= n*p*ceil(m/2) for m even? No, that's not right.
    # Actually, the known bound is: rank(M(n,m,p)) <= n*p*(m - 1 + 1/m) ... no.
    #
    # Let me think differently. The tensor is T[i,j,k] = delta_{j,k} (identity in middle).
    # The rank is the minimum number of rank-1 tensors whose sum gives the identity in the middle.
    #
    # For m=4, we need to decompose the 4x4 identity as a sum of rank-1 matrices.
    # The identity matrix I_4 has rank 4, so we need at least 4 rank-1 matrices.
    # Each rank-1 matrix u_j v_j^T where u_j and v_j are vectors in R^4.
    # Then the full tensor decomposition has rank = n*p * (number of rank-1 terms for I_m).
    # So rank = 2*6*4 = 48? No wait, that's the same as schoolbook.
    #
    # Actually, the schoolbook decomposition already uses m=4 terms for the middle identity,
    # each contributing n*p = 12 terms, for a total of 48.
    #
    # To do better, we need to find a decomposition of the 4x4 identity matrix into
    # fewer than 4 rank-1 matrices? That's impossible since rank(I_4) = 4.
    #
    # Wait, I'm confusing two things. The tensor decomposition of matrix multiplication
    # is NOT just about decomposing the middle identity. Let me reconsider.
    #
    # The matrix multiplication tensor T has T[i,j,k] = 1 if j==k, else 0.
    # A rank-1 tensor is T[i,j,k] = a_i * b_j * c_k.
    # We need sum_t a_i^t * b_j^t * c_k^t = delta_{j,k}.
    #
    # This means: for each j, sum_t b_j^t * c_k^t = delta_{j,k}.
    # And a_i^t can be anything, but we need the full tensor to work.
    #
    # Actually, the standard way: the rank of the matrix multiplication tensor
    # M(n,m,p) is the minimum r such that there exist matrices U (n*m x r), V (m*p x r),
    # W (p*n x r) with the property that the bilinear form gives correct results.
    #
    # For the specific case of multiplying 2x4 by 4x6, the known minimum rank
    # is not trivially less than 48. Let me check small cases.
    #
    # For 1x1*1x1: rank 1.
    # For 1x2*2x1: rank 2 (schoolbook), and it's known that rank is 2.
    # For 2x2*2x2: rank 7 (Strassen), schoolbook is 8.
    #
    # So for 2x4*4x6, can we do better than 48?
    # The general upper bound is n*m*p = 48.
    # Using Strassen-like ideas, for 2x2*2x2 we save 1.
    # For larger matrices, the savings scale.
    #
    # One approach: block decomposition. Write the 4x4 middle as blocks of 2x2.
    # A = [[A11, A12], [A21, A22]] (2x4, so A11, A12 are 2x2)
    # B = [[B11, B12, B13], [B21, B22, B23], [B31, B32, B33], [B41, B42, B43]] 
    #     = [[B_top], [B_bottom]] where B_top is 2x6, B_bottom is 2x6.
    #
    # C = A * B
    # C[i, k] = sum_j A[i,j] * B[j,k]
    #
    # Let me index: A is 2x4, B is 4x6, C is 2x6.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #
    # This is a sum of 4 products. To reduce the number of multiplications,
    # we can try to combine them.
    #
    # For fixed i, C[i,k] is a linear combination of B[0..3,k] with coefficients A[i,0..3].
    # So for each i, we're computing a 1x6 matrix as a linear combination of 4 rows of B.
    # The number of multiplications needed is related to the bilinear complexity.
    #
    # This is equivalent to: given vectors a = (a0,a1,a2,a3) and a 4x6 matrix B,
    # compute a^T * B. The bilinear complexity of this operation is... 
    # It's the rank of the tensor that maps (a, B) to a^T * B.
    #
    # Actually, this is just matrix-vector multiplication (where B is transposed).
    # The rank of the 1x4*4x6 multiplication tensor... 
    #
    # I think the safest approach is to just use the schoolbook decomposition
    # but verify it's correct. The rank is 48.
    
    # Let's just return the schoolbook decomposition which is correct.
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
