"""Rewritten program for matrix multiplication 2,4,8 tensor decomposition."""
import time
import itertools

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Strategy: Exploit the rectangular structure. The schoolbook rank is 2*4*8 = 64.
    We can do better by recognizing that the multiplication is linear in each block.
    
    For C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..1, j in 0..3, k in 0..7.
    
    Key insight: We can use Strassen-like ideas on the 4x8 part.
    Consider the 4x8 matrix B. We can decompose the multiplication A*B where A is 2x4.
    
    Actually, let's think about this differently. The tensor has dimensions 2*4, 4*8, 8*2 = 8, 32, 16.
    Schoolbook rank = 64.
    
    One approach: Group by k. For each k, we have C[:,k] = A * B[:,k] where B[:,k] is 4x1.
    So C[i,k] = sum_j A[i,j] B[j,k].
    
    We can view this as: for each pair (i,k), C[i,k] = dot(A[i,:], B[:,k]).
    
    There are 2*8 = 16 outputs, each is a dot product of length 4.
    
    Can we share computations? The A[i,:] vectors are fixed (2 of them), and B[:,k] are 8 vectors.
    
    Actually, the tensor rank of the bilinear map is what we need.
    
    Let me think about known results. For 2x4 * 4x8:
    - Schoolbook: 64
    - We can potentially use the fact that one dimension is small.
    
    Consider splitting B into two 4x4 blocks: B = [B1 | B2] where B1, B2 are 4x4.
    Then C = [A*B1 | A*B2]. Each A*B is a 2x4 * 4x4 multiplication.
    
    For 2x4 * 4x4: schoolbook rank is 2*4*4 = 32.
    Can we do better for 2x4 * 4x4?
    
    For 2x4 * 4x4, C[i,k] = sum_j A[i,j] B[j,k], i=0,1; j=0..3; k=0..3.
    
    One idea: Use the identity that for 2x4 * 4x4, we can write it as a product involving
    smaller matrices. 
    
    Actually, let's consider the rank of the tensor. The tensor T is in R^{8} ⊗ R^{32} ⊗ R^{16}
    where the three modes correspond to (i,j), (j,k), (k,i) respectively.
    
    T_{(i,j),(j,k),(k,i)} = 1 if indices match, 0 otherwise.
    
    For the 2x4*4x8 case:
    - Mode 1: (i,j) with i in 0..1, j in 0..3 → 8 elements
    - Mode 2: (j,k) with j in 0..3, k in 0..7 → 32 elements  
    - Mode 3: (k,i) with k in 0..7, i in 0..1 → 16 elements
    
    Schoolbook rank = 64.
    
    Let me try a different approach. Since n=2 is small, we can enumerate possible decompositions.
    
    For a rank-r decomposition, we need:
    U: 8 x r, V: 32 x r, W: 16 x r
    such that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    Wait, that's not right. The tensor entry T[(i,j),(j,k),(k,i)] should equal 
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And T[(i,j),(j',k),(k',i)] = 0 if j != j' or k != k'.
    
    Actually the standard form is:
    T[(i,j),(j',k),(k',i)] = delta_{j,j'} * delta_{k,k'}
    
    And we need: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i, c] = delta_{j,j'} * delta_{k,k'}
    
    This is a complex constraint. Let me just implement the schoolbook and see if I can find improvements.
    
    Actually, for 2x4*4x8, let me try to use the fact that we can combine some terms.
    
    Consider: C[i,k] = sum_j A[i,j] B[j,k]
    
    For fixed i, this is a linear combination of the rows of B weighted by A[i,:].
    
    Since there are only 2 rows of A, we have:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    This is 16 bilinear forms, each of degree (1,1) in the variables.
    
    The tensor rank is the minimum number of rank-1 tensors needed.
    
    For now, let me just return the schoolbook decomposition. The rank is 64.
    """
    n, m, p = DIMENSIONS
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
