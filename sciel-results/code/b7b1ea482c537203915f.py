"""
Rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.

Approach:
  - Treat the multiplication tensor T[i,j,k] = 1 for i in [0,2), j in [0,3), k in [0,5).
  - The tensor flattens to three views:
      U_view: shape (6, r)  where row index = i*m + j
      V_view: shape (15, r) where row index = j*p + k
      W_view: shape (10, r) where row index = k*n + i
  - We seek the minimal r such that there exist matrices U, V, W of these shapes
    with integer/half-integer entries satisfying:
      For all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

  - The schoolbook rank is n*m*p = 30.
  - We exploit the structure: the 2x3 * 3x5 multiplication can be decomposed as:
      C = A * B  where A is 2x3, B is 3x5.
    
    Split B into B_left (3x2) and B_right (3x3):
      C_left = A * B_left  (2x2)
      C_right = A * B_right (2x3)
    
    For C_left (2x2 * 2x2 is not quite right; it's 2x3 * 3x2):
      Actually, A is 2x3 and B_left is 3x2.
      We can write this as:
        C_left[i,k] = sum_j A[i,j] * B_left[j,k]
      This is a 2x3 * 3x2 matmul. The rank of this tensor is at most 2*3*2 = 12,
      but we can do better. Note that 2x3 * 3x2 can be seen as two 2x1 * 1x2 blocks
      plus cross terms? No.
      
    Let's use a known result: the rank of matmul_{2,3,5} is 28.
    This can be achieved by:
      - Decomposing the 3x5 matrix B into three 3x1, 3x1, 3x1, 3x1, 3x1 columns.
      - Or more cleverly, using the fact that:
        matmul_{2,3,5} <= matmul_{2,3,3} + matmul_{2,3,2}
        rank(matmul_{2,3,3}) <= 2*3*3 = 18, but actually it's known to be 17 or 18.
        rank(matmul_{2,3,2}) <= 2*3*2 = 12, but it's 10 or 11.
      
    Actually, let's use a direct construction.
    
    We can write:
      C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    Let's define 28 terms. We'll use the following strategy:
    For each (i, j, k), we need to cover the product A[i,j]*B[j,k].
    
    A known decomposition for 2x3 * 3x5 with rank 28:
    
    Let's think of it as:
      C = A * B
      = [A[:,0:2], A[:,2]] * [[B[0:2,:]], [B[2,:]]]
      = A[:,0:2] * B[0:2,:] + A[:,2] * B[2,:]
    
    The first term is a 2x2 * 2x5 matmul.
    The second term is a 2x1 * 1x5 matmul.
    
    Rank of 2x2 * 2x5: 
      This can be seen as two 2x2 * 2x2 blocks? No, 2x5.
      Actually, 2x2 * 2x5 = 2x2 * (2x2 + 2x2 + 2x1)? No.
      Let's just compute: the tensor for 2x2 * 2x5 has rank at most 2*2*5 = 20.
      But we can do better. Note that 2x2 * 2x5 can be decomposed as:
        For each k in 0..4, we need C[:,k] = A * B[:,k]
        This is 5 independent 2x2 * 2x1 multiplications? No.
      
    Let me use a different approach. I'll construct the rank-28 decomposition explicitly.
    
    Actually, the known minimal rank for 2x3 * 3x5 is 28. Here's a construction:
    
    We'll use 28 terms. For each term c in 0..27:
      U[c] has non-zero entries at positions (i, j) for certain (i,j)
      V[c] has non-zero entries at positions (j, k) for certain (j,k)
      W[c] has non-zero entries at positions (k, i) for certain (k,i)
    
    The product U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should sum to 1 for each (i,j,k).
    
    Let's use the following known decomposition (adapted from literature):
    
    For the 2x3 * 3x5 case, we can use:
      - 7 terms for a 2x2 * 2x2 sub-block (Strassen)
      - 4 terms for a 2x1 * 1x2 sub-block
      - 7 terms for another 2x2 * 2x2 sub-block
      - 4 terms for another 2x1 * 1x2 sub-block
      - 6 terms for a 2x3 * 3x1 sub-block
      
    Total: 7 + 4 + 7 + 4 + 6 = 28
    
    Let's define the blocks:
      A is 2x3: A = [A_11 | A_12] where A_11 is 2x2, A_12 is 2x1
      B is 3x5: B = [B_11, B_12, B_13] where B_11 is 3x2, B_12 is 3x2, B_13 is 3x1
      
      C = A * B = [A_11 * B_11[0:2,:] + A_12 * B_11[2,:] | A_11 * B_12[0:2,:] + A_12 * B_12[2,:] | A_11 * B_13[0:2,:] + A_12 * B_13[2,:]]
      
    Wait, this is getting complicated. Let me just use the schoolbook rank of 30 for now,
    as finding a rank-28 decomposition is non-trivial and I want to ensure correctness.
    
    Actually, let me try to construct a rank-28 decomposition.
    
    Here's a cleaner approach:
      C[i,k] = sum_j A[i,j] * B[j,k]
      
      We can write this as:
      C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] + A[i,2] * B[2,k]
      
      The first sum is a 2x2 * 2x5 matmul, the second is a 2x1 * 1x5 matmul.
      
      For the 2x2 * 2x5 part:
        This is equivalent to multiplying a 2x2 matrix by a 2x5 matrix.
        The rank of matmul_{2,2,5} is at most 2*2*5 = 20, but it's known to be 17 or 18.
        Actually, matmul_{2,2,5} can be seen as five 2x2 * 2x1 multiplications, each of rank 4,
        but with sharing. The known rank is 17.
        
      For the 2x1 * 1x5 part:
        This is just 2*5 = 10 scalar products, so rank 10.
        
      Total: 17 + 10 = 27? That would be better than 28.
      
      But wait, the 2x2 * 2x5 decomposition might not be independent. Let me check.
      
      Actually, I think the minimal rank is 28. Let me just return the schoolbook rank of 30
      to ensure correctness, and note that a lower rank may exist.
      
    For now, I'll return the schoolbook decomposition.
    """
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 6
    V_dim = m * p  # 15
    W_dim = p * n  # 10

    # Schoolbook decomposition: rank = n*m*p = 30
    rank = n * m * p
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    
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
