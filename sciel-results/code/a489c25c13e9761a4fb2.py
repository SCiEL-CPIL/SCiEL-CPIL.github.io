"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x4 matrix by a 4x4 matrix.
    
    The schoolbook rank is 2*4*4 = 32.
    We attempt to find a lower rank decomposition.
    
    Strategy:
    1. First, try to find a rank-24 decomposition using a rectangular Winograd-like approach.
       The idea is to exploit the structure where we multiply two rows of A (2x4) by the same B (4x4).
       
    2. If that fails, fall back to schoolbook rank 32.
    
    For the 2x4x4 case, we can think of it as:
    C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    
    One approach: group the j-index. For each pair of j's, we can use a 2x2x4 sub-problem.
    But a more direct approach for 2x4x4:
    
    Consider that we need to compute 2*4 = 8 output values per column of B, for 4 columns.
    
    Let's try a decomposition based on the fact that for each k (output column), we compute:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    This is essentially two dot products of length 4.
    
    A known result: the rank of matrix multiplication for (n,m,p) is at most n*m*p,
    but can be lower. For 2x4x4, let's see if we can do better than 32.
    
    Actually, let's use a simple but effective approach: 
    We can view the 2x4x4 multiplication as a block structure.
    
    Let's try rank 24 first. The idea: for each of the 4 columns of B, we need to compute
    two linear combinations of the 4 entries of A's rows. 
    
    Alternative: use the fact that 2x4x4 can be decomposed using 4 "blocks" of 2x1x4 or similar.
    
    Actually, a cleaner approach: 
    The tensor is T[i,j,k] = delta_{i,k} (no, that's not right).
    
    Let me think differently. The matrix multiplication tensor for (n,m,p) has entries:
    T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    and the multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    
    For n=2, m=4, p=4:
    We have 2*4*4 = 32 terms in the schoolbook.
    
    Can we do better? Let's try a rank-28 or rank-24 decomposition.
    
    One known technique: for multiplying a (2x4) by (4x4), we can use the following:
    Group the 4 columns of B into pairs, and use a 2x2x2 Strassen-like decomposition for each pair?
    
    Actually, let's just implement the schoolbook rank 32 first, and then try to optimize.
    But the task asks for rank-minimal, so we should try to find a lower rank.
    
    Let me try a specific construction for rank 24:
    
    For each k in {0,1,2,3}, we need to compute:
    C[0,k] = sum_{j=0}^3 A[0,j] B[j,k]
    C[1,k] = sum_{j=0}^3 A[1,j] B[j,k]
    
    This is equivalent to multiplying a 2x4 matrix A by a 4x1 vector (column k of B).
    
    The rank of the 2x4x1 multiplication is at most 2*4 = 8, but can it be lower?
    For 2x4x1, the tensor is T[i,j,0] = delta_{i,0} (no).
    
    Actually, for 2x4x1, we're computing:
    C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]
    C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0] + A[1,3]*B[3,0]
    
    This is a 2x4x1 tensor. The rank of this tensor is at most 8 (schoolbook).
    
    If we could decompose each 2x4x1 with rank 6, then total rank would be 4*6 = 24.
    
    Is the rank of 2x4x1 at most 6? Let's see.
    
    The 2x4x1 tensor can be viewed as two linear forms in B, each being a linear combination of A's entries.
    
    Actually, I think the minimal rank for 2x4x1 is 4, not 6. Here's why:
    We can write:
    C[0] = (A[0,0]+A[1,0]) * B[0,0] + (A[0,1]+A[1,1]) * B[1,0] + ... 
    No, that doesn't work directly.
    
    Let me just go with schoolbook rank 32 for now, as it's guaranteed to work.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 32
    
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
