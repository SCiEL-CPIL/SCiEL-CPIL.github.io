import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _strassen_222():
    """
    Return the standard Strassen decomposition of the 2x2x2 tensor.
    U, V, W are each 4x7 matrices with entries in {-1, 0, 1}.
    Tensor T[i*2+j, j*2+k, k*2+i] = sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r]
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # m1 = (a0+a3)*(b0+b3)
    # m2 = (a2+a3)*b0
    # m3 = a0*(b1-b2)
    # m4 = a1*(b3-b2)
    # m5 = (a0+a1)*b2
    # m6 = (a3+a1)*(b1+b3)
    # m7 = (a2+a0)*(b0+b1)
    #
    # C[0,0] = m1 + m4 - m5 - m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Index mapping: U[i*2+j, r] corresponds to coefficient of A[i,j] in product r
    # V[j*2+k, r] corresponds to coefficient of B[j,k] in product r
    # W[k*2+i, r] corresponds to coefficient of C[i,k] in product r
    
    # m1: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    U[0, 0] = 1  # A[0,0]
    U[3, 0] = 1  # A[1,1]
    V[0, 0] = 1  # B[0,0]
    V[3, 0] = 1  # B[1,1]
    W[0, 0] = 1  # C[0,0]
    W[3, 0] = 1  # C[1,1]
    
    # m2: (A[1,0]+A[1,1])*B[0,0]
    U[2, 1] = 1  # A[1,0]
    U[3, 1] = 1  # A[1,1]
    V[0, 1] = 1  # B[0,0]
    W[2, 1] = 1  # C[1,0]
    W[3, 1] = -1 # C[1,1]
    
    # m3: A[0,0]*(B[0,1]-B[1,1])
    U[0, 2] = 1  # A[0,0]
    V[1, 2] = 1  # B[0,1]
    V[3, 2] = -1 # B[1,1]
    W[1, 2] = 1  # C[0,1]
    W[3, 2] = 1  # C[1,1]
    
    # m4: A[0,1]*(B[1,1]-B[1,0])
    U[1, 3] = 1  # A[0,1]
    V[3, 3] = 1  # B[1,1]
    V[2, 3] = -1 # B[1,0]
    W[0, 3] = 1  # C[0,0]
    W[2, 3] = 1  # C[1,0]
    
    # m5: (A[0,0]+A[0,1])*B[1,0]
    U[0, 4] = 1  # A[0,0]
    U[1, 4] = 1  # A[0,1]
    V[2, 4] = 1  # B[1,0]
    W[0, 4] = -1 # C[0,0]
    W[1, 4] = 1  # C[0,1]
    
    # m6: (A[1,1]+A[0,1])*(B[0,1]+B[1,1])
    U[3, 5] = 1  # A[1,1]
    U[1, 5] = 1  # A[0,1]
    V[1, 5] = 1  # B[0,1]
    V[3, 5] = 1  # B[1,1]
    W[3, 5] = 1  # C[1,1]
    
    # m7: (A[1,0]+A[0,0])*(B[0,0]+B[0,1])
    U[2, 6] = 1  # A[1,0]
    U[0, 6] = 1  # A[0,0]
    V[0, 6] = 1  # B[0,0]
    V[1, 6] = 1  # B[0,1]
    W[0, 6] = -1 # C[0,0]
    
    return U, V, W

def construct():
    n, m, p = 2, 4, 8
    
    # Decompose into 8 blocks of 2x2x2
    # The 2x4x8 multiplication can be viewed as:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # We group the 4 middle dimensions into 2 pairs: (0,1) and (2,3)
    # For each output column k (0..7), we have:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #
    # The key insight is that this is equivalent to 8 independent 2x2x2 multiplications
    # when we consider appropriate blockings, but more precisely, we can use the
    # Strassen decomposition on 2x2 submatrices.
    
    # Actually, a cleaner approach: the 2x4x8 tensor can be decomposed into
    # 8 copies of the 2x2x2 tensor. Here's how:
    # For each pair of middle indices (j1, j2) in {(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)}
    # No, that's not right either.
    
    # Let me think differently. The total rank is 8 * 7 = 56 according to the knowledge.
    # The 2x4x8 problem decomposes into 8 blocks of 2x2x2.
    
    # One way to see this: split the 4 columns of A into two groups of 2,
    # and the 8 rows of B correspondingly. But the rows of B are independent.
    
    # Actually, the correct decomposition is:
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # This is a sum of 4 terms for each (i,k).
    # 
    # If we had 2x2x8, we could use Strassen on the 2x2 part.
    # For 2x4x8, we can split into two 2x2x8 problems and add them.
    # Each 2x2x8 has rank 2*8 + something? No.
    
    # Let me use a different approach. The rank of the 2x4x8 tensor is at most
    # 8 * rank(2x2x2) = 8 * 7 = 56, by viewing it as 8 independent 2x2x2 multiplications.
    # 
    # Here's the explicit decomposition:
    # For each k in 0..7, consider the 2x4 submatrix A and the 4x1 column B[:,k].
    # C[:,k] = A * B[:,k]
    # This is a 2x4x1 multiplication, which has rank 4 (schoolbook) or can be
    # optimized. But 2x4x1 is just 8 scalar multiplications with additions.
    # 
    # The tensor rank of 2x4x1 is 4 (since it's just matrix-vector multiply,
    # which requires 4 independent multiplications for each of 2 outputs, but
    # shared). Actually 2x4x1 has rank 4.
    # So total rank would be 8*4 = 32? But that's not how tensor rank works
    # for the full tensor.
    
    # Let me just implement the 8-block Strassen decomposition directly.
    # The idea: partition the middle dimension into 2 groups of 2.
    # For group g in {0, 1} (indices 2g, 2g+1):
    #   We have A[:, 2g:2g+2] (2x2) and B[2g:2g+2, :] (2x8)
    #   The contribution is A[:, 2g:2g+2] @ B[2g:2g+2, :]
    #   This is a 2x2x8 multiplication.
    
    # For 2x2x8, we can apply Strassen in the 2x2 dimensions:
    # C[:,k] = (2x2 matrix) @ (2x1 vector) for each k
    # Using Strassen: 7 multiplications per output vector, but the 7 products
    # depend on k through B's second dimension.
    
    # Actually, for 2x2xn, the tensor rank is known. Let me just use
    # 8 independent 2x2x2 blocks.
    
    # The cleanest decomposition: For each of the 8 "slots" corresponding to
    # (j, k) pairs where we apply Strassen, but that's not quite right.
    
    # Let me use the fact that the 2x4x8 tensor is the direct sum of 8 copies
    # of a 2x2x2 tensor under a suitable basis change.
    
    # Explicitly: define 8 rank-1 tensors that sum to the multiplication tensor.
    # T[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # where U is (2*4, R), V is (4*8, R), W is (8*2, R)
    
    # U[i*4+j, r], V[j*8+k, r], W[k*2+i, r]
    
    # I'll construct this by tiling the Strassen 2x2x2 decomposition.
    # Split j into j_hi (0 or 1) and j_lo (0,1,2,3 -> pair (j_lo, j_hi))
    # Actually, let me just enumerate 8 blocks.
    
    # Block b in 0..7: corresponds to a specific (j1, j2, k_offset) pattern.
    
    # Simpler: use the 8 rank-7 Strassen decompositions, one for each
    # combination of output column group and middle pair.
    
    # Let me just build it by blocks.
    # For each k in 0..7, and for the 2x4 part, we can write:
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # 
    # Group j=0,1 together and j=2,3 together.
    # C[i,k] = (A[i,0]B[0,k] + A[i,1]B[1,k]) + (A[i,2]B[2,k] + A[i,3]B[3,k])
    # 
    # The first term is a 2x2x1 multiply (for fixed k), rank 3 (Strassen)
    # The second term is another 2x2x1 multiply, rank 3
    # Total per k: 6, times 8 = 48? But these share structure.
    
    # I think the safest bet is to just implement the known rank-56 decomposition.
    # Let me construct it explicitly using 8 Strassen blocks.
    
    U_str, V_str, W_str = _strassen_222()
    
    # The 2x4x8 tensor decomposes as follows:
    # We have 4 middle indices. Pair them as (0,1) and (2,3).
    # For pair (0,1): A[:,0:2] is 2x2, B[0:2,:] is 2x8
    #   This gives a 2x2x8 sub-tensor.
    # For pair (2,3): A[:,2:4] is 2x2, B[2:4,:] is 2x8
    #   This gives another 2x2x8 sub-tensor.
    
    # For each 2x2x8 sub-tensor, we can decompose it as 4 copies of 2x2x2
    # (one for each pair of output columns? No, 2x2x8 means 2x2 input, 8 outputs).
    # 
    # Actually, 2x2x8: C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] for k=0..7
    # This is 8 independent 2x2x1 multiplications.
    # Rank of 2x2x1 is 3 (using Strassen-like).
    # So rank of 2x2x8 is at most 8*3 = 24? But we can do better by sharing.
    
    # Hmm, I'm going in circles. Let me just use the schoolbook for the inner
    # 2x2 and see if I can get rank 56.
    
    # Direct approach: The 2x4x8 tensor has 64 "terms" in the sum.
    # Each term is A[i,j] * B[j,k] contributing to C[i,k].
    # Using Strassen on 2x2 blocks: for each pair of j's and each pair of k's,
    # we save 1 multiplication.
    
    # There are 2 choices for j-pair (out of 4, choose 2 = 6 pairs) and
    # 4 choices for k-pair (out of 8, choose 2 = 28 pairs). That's too many.
    
    # The knowledge says rank 56, so 8 savings from 64.
    # 8 savings = 8 Strassen applications on 2x2x2 blocks.
    
    # Let me just implement 8 Strassen blocks explicitly.
    # Block r in 0..7: uses middle indices and output indices in a specific pattern.
    
    # I'll define the 8 blocks as:
    # Block r: j indices are (r//4, r//4 + 1) if we pair them, and k indices...
    
    # Actually, let me just do: for each k in 0..7, apply a 2x4x1 decomposition.
    # But 2x4x1 has rank 4, giving 32 total. Let me verify if rank 32 works.
    
    # 2x4x1: C[i] = sum_{j=0}^{3} A[i,j] B[j,0]
    # This is just two dot products of length 4.
    # Rank is 4 (each of the 4 A[i,j] * B[j,0] products is independent).
    # So 8 * 4 = 32.
    
    # But wait, the tensor rank of the full 2x4x8 tensor is not the sum of
    # ranks of slices. It's at most the sum, but could be less.
    
    # Let me try rank 32 with a direct construction.
    
    rank = 32
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    for k in range(p):
        for j in range(m):
            # This is schoolbook: rank = n*m*p = 64
            # But I want rank 32, so I need to combine.
            pass
    
    # Let me just do schoolbook first to make sure it's correct, then optimize.
    # Schoolbook gives rank 64.
    
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
