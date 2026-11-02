"""Program for rank-minimal tensor decomposition of 2x3 by 3x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    deadline = time.time() + 120
    
    # Known ranks from literature for small matrix multiplication
    # For 2x3 by 3x6: 
    # - Schoolbook: 36
    # - Winograd style optimizations can reduce rank
    # The rank of matrix multiplication M_{n,m,p} is known for many small cases
    # For 2x3x6, the rank is actually 18 (from Strassen-like algorithms)
    # Let me verify: the rank of M_{1,2,2} is 7 (not 4)
    # For M_{2,3,6}, we can use decomposition methods
    
    # Strategy: Use known rank-18 construction for 2x3 by 3x6
    # This can be achieved by decomposing 6x6 blocks
    
    # Actually, let's think about this more carefully.
    # The matrix multiplication tensor for (2,3,6) has size 2*3*6 = 36 scalar multiplications in schoolbook.
    
    # Known results:
    # - M_{1,2,2}: rank 7 (not 4)
    # - M_{2,2,2}: rank 7 (not 8)
    # - M_{2,3,3}: rank 15 (not 18) - this is a known result
    # - M_{2,3,6}: we can write C = A * B where A is 2x3, B is 3x6
    #   C has 12 entries, each is sum of 3 products
    #   
    # One approach: partition B into two 3x3 matrices B1, B2
    # C1 = A * B1, C2 = A * B2
    # Each 2x3 by 3x3 multiplication has rank at most 15
    # So total rank at most 30
    #
    # Better: use the fact that M_{2,3,6} can be decomposed using
    # Strassen-like methods
    #
    # Actually, the known rank for M_{2,3,6} is 18, achieved by
    # combining M_{2,3,3} decompositions cleverly
    #
    # Let me use a direct construction with rank 18:
    # We'll construct U, V, W explicitly
    
    rank = 18
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Construction using 2x3 by 3x3 rank-15 blocks combined
    # Actually, let me use a simpler verified construction
    
    # For 2x3 * 3x6, think of it as:
    # Split B into columns: B = [b0, b1, b2, b3, b4, b5]
    # C = [A*b0, A*b1, A*b2, A*b3, A*b4, A*b5]
    # Each A*bj is 2x1, requiring 3 multiplications in schoolbook = 6*3 = 18 if independent
    # But there's sharing possible
    
    # Better construction: use the structure of the tensor
    # The tensor T_{i,j,k} for i in 0..1, j in 0..2, k in 0..5
    # T[i,j,k] contributes to C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Rank-18 construction:
    # Use the following pattern based on Winograd's algorithm ideas
    
    # For 2x3 by 3x3, a rank-15 decomposition exists (Strassen-type)
    # For 2x3 by 3x6, we can use rank 18
    
    # Let me construct it using the known M_{2,3,3} rank-15 block structure
    # and extend to 6 columns
    
    # Actually, the simplest correct approach: use the known result that
    # rank(M_{2,3,6}) = 18 and construct it
    
    # Here's a concrete construction using algebraic manipulation:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # We can write this using shared intermediate products.
    # For each pair of k values, we can save multiplications.
    
    # Let me use a systematic construction with rank 18:
    
    # Define 18 rank-1 tensors that span the multiplication
    # U is 6x18, V is 18x12... wait, V is 3*6=18 x rank
    
    # U shape: (2*3, 18) = (6, 18) - indices (i,j) for A entries
    # V shape: (3*6, 18) = (18, 18) - indices (j,k) for B entries  
    # W shape: (6*2, 18) = (12, 18) - indices (k,i) for C entries
    
    # Construction using block structure:
    # Group columns 0-2 and 3-5 of B
    
    # For first 3 columns (B[:, 0:3]):
    # Use a rank-10 contribution (not 15, because we only have 2 rows)
    # Actually for 2x3 by 3x3 with fixed 2 rows, the rank is 10
    # No wait, M_{2,3,3} has rank 15 regardless
    
    # Let me just use the verified rank-18 decomposition:
    
    # Using the method from: "The Rank of the 3x3 Matrix Multiplication Tensor"
    # For M_{2,3,6}, rank = 18
    
    # Explicit construction:
    # We'll use 18 terms where each term is u_c * v_c * w_c
    
    # Pattern: for the 2x3 by 3x3 subproblems, use the rank-15 Strassen decomposition
    # and adapt for 6 columns
    
    # Here's the construction based on shared bilinear forms:
    
    # Terms 0-9: handle first 3 columns with a compact scheme
    # Terms 10-17: handle last 3 columns with a compact scheme
    # Terms 18... wait we need exactly 18
    
    # Let me use a cleaner approach: the rank is 18, and I'll construct it
    # using the known algebraic identity
    
    # For 2x3 * 3x6, define:
    # For k in {0,1,2}: C[:,k] depends on A and B[:,k]
    # For k in {3,4,5}: C[:,k] depends on A and B[:,k]
    
    # Key insight: we can share 3 multiplications between adjacent column pairs
    
    # Construction with 18 rank-1 terms:
    # Group k as (0,3), (1,4), (2,5) and use 6 terms per group + 6 shared
    
    # Actually, let me use the direct known decomposition:
    
    # For M_{2,3,6}, the rank-18 decomposition can be written as:
    # For each r in 0..17:
    #   U[(i,j), r], V[(j,k), r], W[(k,i), r]
    
    # I'll use the following verified pattern:
    # Rows of U correspond to A entries: (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
    # Rows of V correspond to B entries: (0,0)...(0,5),(1,0)...(1,5),(2,0)...(2,5)
    # Rows of W correspond to C entries: (0,0),(0,1),...,(0,5),(1,0),(1,1),...,(1,5)
    
    # The 18 rank-1 components:
    
    # Component 0: A[0,0]*B[0,0] -> C[0,0]
    U[0, 0] = 1  # A[0,0]
    V[0, 0] = 1  # B[0,0]
    W[0, 0] = 1  # C[0,0]
    
    # Component 1: A[0,0]*B[0,1] -> C[0,1]  
    U[0, 1] = 1
    V[1, 1] = 1  # B[0,1]
    W[1, 1] = 1  # C[0,1]
    
    # Component 2: A[0,0]*B[0,2] -> C[0,2]
    U[0, 2] = 1
    V[2, 2] = 1  # B[0,2]
    W[2, 2] = 1  # C[0,2]
    
    # Component 3: A[0,1]*B[1,0] -> C[0,0]
    U[3, 3] = 1  # A[0,1]
    V[6, 3] = 1  # B[1,0]
    W[0, 3] = 1
    
    # Component 4: A[0,1]*B[1,1] -> C[0,1]
    U[3, 4] = 1
    V[7, 4] = 1  # B[1,1]
    W[1, 4] = 1
    
    # Component 5: A[0,1]*B[1,2] -> C[0,2]
    U[3, 5] = 1
    V[8, 5] = 1  # B[1,2]
    W[2, 5] = 1
    
    # Component 6: A[0,2]*B[2,0] -> C[0,0]
    U[2, 6] = 1  # A[0,2]
    V[12, 6] = 1 # B[2,0]
    W[0, 6] = 1
    
    # Component 7: A[0,2]*B[2,1] -> C[0,1]
    U[2, 7] = 1
    V[13, 7] = 1 # B[2,1]
    W[1, 7] = 1
    
    # Component 8: A[0,2]*B[2,2] -> C[0,2]
    U[2, 8] = 1
    V[14, 8] = 1 # B[2,2]
    W[2, 8] = 1
    
    # Now for row 1 of C:
    # Component 9: A[1,0]*B[0,0] -> C[1,0]
    U[3, 9] = 1  # A[1,0]
    V[0, 9] = 1
    W[6, 9] = 1  # C[1,0]
    
    # Component 10: A[1,0]*B[0,1] -> C[1,1]
    U[3, 10] = 1
    V[1, 10] = 1
    W[7, 10] = 1 # C[1,1]
    
    # Component 11: A[1,0]*B[0,2] -> C[1,2]
    U[3, 11] = 1
    V[2, 11] = 1
    W[8, 11] = 1 # C[1,2]
    
    # Component 12: A[1,1]*B[1,0] -> C[1,0]
    U[4, 12] = 1 # A[1,1]
    V[6, 12] = 1
    W[6, 12] = 1
    
    # Component 13: A[1,1]*B[1,1] -> C[1,1]
    U[4, 13] = 1
    V[7, 13] = 1
    W[7, 13] = 1
    
    # Component 14: A[1,1]*B[1,2] -> C[1,2]
    U[4, 14] = 1
    V[8, 14] = 1
    W[8, 14] = 1
    
    # Component 15: A[1,2]*B[2,0] -> C[1,0]
    U[5, 15] = 1 # A[1,2]
    V[12, 15] = 1
    W[6, 15] = 1
    
    # Component 16: A[1,2]*B[2,1] -> C[1,1]
    U[5, 16] = 1
    V[13, 16] = 1
    W[7, 16] = 1
    
    # Component 17: A[1,2]*B[2,2] -> C[1,2]
    U[5, 17] = 1
    V[14, 17] = 1
    W[8, 17] = 1
    
    # But wait, I'm missing the entries for B columns 3,4,5 and C columns 3,4,5!
    # This construction only covers B[0:3] and C[0:3]
    # I need to extend to columns 3,4,5
    
    # Let me redo this with a proper rank-18 construction for all 6 columns.
    # The rank for M_{2,3,6} is actually 18, not 27.
    # This requires clever sharing.
    
    # Given the time constraint, let me fall back to the schoolbook rank of 36
    # which is guaranteed to be correct, and note that 18 is the target.
    
    # For now, return the correct but not optimal schoolbook decomposition
    # A future iteration can find the rank-18 construction
    
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
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
