import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill

    n, m, p = 3, 5, 5
    total_dim = n * m * p
    U_dim = n * m  # 15
    V_dim = m * p  # 25
    W_dim = p * n  # 15

    # We will build the decomposition using a combination of:
    # 1. Strassen-like block decomposition for 2x2 sub-blocks where possible
    # 2. Schoolbook for the remaining terms
    #
    # Strategy:
    # Partition A (3x5) into blocks: A = [A11, A12; A21, A22; A31, A32]
    # where A11, A21, A31 are 1x2, A12, A22, A32 are 1x3
    # Partition B (5x5) into blocks: B = [B11, B12; B21, B22]
    # where B11, B21 are 2x2, B12, B22 are 2x3
    #
    # Actually, let's use a simpler approach:
    # Use the known result that 3x5x5 can be done in rank 39.
    # But let's first try to get a good decomposition by combining Strassen blocks.
    #
    # Let me think about this more carefully.
    #
    # A is 3x5, B is 5x5, C is 3x5.
    #
    # One approach: partition the columns of A and rows of B into 2+3 or 1+2+2, etc.
    #
    # Let's try: partition the middle dimension (5) into 2+3.
    # A = [A_left | A_right] where A_left is 3x2, A_right is 3x3
    # B = [B_top; B_bottom] where B_top is 2x5, B_bottom is 3x5
    #
    # C = A_left @ B_top + A_right @ B_bottom
    #
    # A_left @ B_top is 3x2 @ 2x5 = 3x5
    # A_right @ B_bottom is 3x3 @ 3x5 = 3x5
    #
    # For A_left @ B_top: 3x2x5. Rank?
    # For A_right @ B_bottom: 3x3x5. Rank?
    #
    # Total rank = rank(3x2x5) + rank(3x3x5)
    #
    # 3x2x5: schoolbook is 30. Can we do better?
    # 3x3x5: schoolbook is 45. Can we do better?
    #
    # Let's try a different partition.
    #
    # Actually, let me just implement a direct search using known Strassen-based
    # block decompositions.
    #
    # Known: 2x2x2 has rank 7 (Strassen)
    # 2x2x1 has rank 4 (schoolbook, can't do better for rectangular)
    # 1x1x1 has rank 1
    #
    # Let me try to build the decomposition by partitioning the 3x5x5 tensor
    # into blocks and applying Strassen where we have 2x2x2 sub-problems.
    #
    # Partition:
    # i dimension (3): split into 2+1
    # j dimension (5): split into 2+2+1
    # k dimension (5): split into 2+2+1
    #
    # This gives 2*3*3 = 18 blocks.
    #
    # For blocks where i-part=2, j-part=2, k-part=2: use Strassen (rank 7 instead of 8)
    # For other blocks: use schoolbook.
    #
    # Let me count:
    # i-part=2 (first 2 rows), j-part=2 (first 2 or next 2 cols), k-part=2 (first 2 or next 2 rows of B)
    #
    # Blocks:
    # (i=0-1, j=0-1, k=0-1): 2x2x2 -> Strassen, rank 7
    # (i=0-1, j=0-1, k=2-3): 2x2x2 -> Strassen, rank 7
    # (i=0-1, j=2-3, k=0-1): 2x2x2 -> Strassen, rank 7
    # (i=0-1, j=2-3, k=2-3): 2x2x2 -> Strassen, rank 7
    # (i=0-1, j=4, k=0-1): 2x1x2 -> schoolbook, rank 4
    # (i=0-1, j=4, k=2-3): 2x1x2 -> schoolbook, rank 4
    # (i=0-1, j=4, k=4): 2x1x1 -> schoolbook, rank 2
    # (i=0-1, j=0-1, k=4): 2x2x1 -> schoolbook, rank 4
    # (i=0-1, j=2-3, k=4): 2x2x1 -> schoolbook, rank 4
    # (i=2, j=0-1, k=0-1): 1x2x2 -> schoolbook, rank 4
    # (i=2, j=0-1, k=2-3): 1x2x2 -> schoolbook, rank 4
    # (i=2, j=2-3, k=0-1): 1x2x2 -> schoolbook, rank 4
    # (i=2, j=2-3, k=2-3): 1x2x2 -> schoolbook, rank 4
    # (i=2, j=4, k=0-1): 1x1x2 -> schoolbook, rank 2
    # (i=2, j=4, k=2-3): 1x1x2 -> schoolbook, rank 2
    # (i=2, j=4, k=4): 1x1x1 -> schoolbook, rank 1
    # (i=2, j=0-1, k=4): 1x2x1 -> schoolbook, rank 2
    # (i=2, j=2-3, k=4): 1x2x1 -> schoolbook, rank 2
    #
    # Total rank = 4*7 + 4+4+2+4+4 + 4+4+4+4+2+2+1+2+2
    # = 28 + 18 + 23 = 69
    #
    # That's not great. Let me try a different partition.
    #
    # Let me try: i: 2+1, j: 2+3, k: 2+3
    # Blocks: 2*2*2 = 8
    # (i=0-1, j=0-1, k=0-1): 2x2x2 -> Strassen, 7
    # (i=0-1, j=0-1, k=2-4): 2x2x3 -> schoolbook, 12
    # (i=0-1, j=2-4, k=0-1): 2x3x2 -> schoolbook, 12
    # (i=0-1, j=2-4, k=2-4): 2x3x3 -> schoolbook, 18
    # (i=2, j=0-1, k=0-1): 1x2x2 -> schoolbook, 4
    # (i=2, j=0-1, k=2-4): 1x2x3 -> schoolbook, 6
    # (i=2, j=2-4, k=0-1): 1x3x2 -> schoolbook, 6
    # (i=2, j=2-4, k=2-4): 1x3x3 -> schoolbook, 9
    #
    # Total = 7+12+12+18+4+6+6+9 = 74. Worse.
    #
    # Let me try to be smarter. The key insight from the measured result is
    # that using Strassen for 2x2x2 sub-blocks reduces rank from 75 to 71.
    # That means only one 2x2x2 block was replaced.
    #
    # Let me try: i: 2+1, j: 2+2+1, k: 2+2+1
    # And only use Strassen for the (2,2,2) blocks.
    #
    # Actually, let me just implement a systematic approach:
    # Try all reasonable partitions and compute the rank.
    #
    # For now, let me implement the partition i:2+1, j:2+2+1, k:2+2+1
    # with Strassen on 2x2x2 blocks.
    
    # Let me build this manually.
    # 
    # Partitions:
    # i: [0,1], [2]
    # j: [0,1], [2,3], [4]
    # k: [0,1], [2,3], [4]
    #
    # I'll collect all (U, V, W) triples for each block.
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        U_rows.append((u_idx, u_val))
        V_rows.append((v_idx, v_val))
        W_rows.append((w_idx, w_val))
    
    def add_block_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        """Add schoolbook terms for a block."""
        for i in range(i_start, i_end + 1):
            for j in range(j_start, j_end + 1):
                for k in range(k_start, k_end + 1):
                    add_term(i * m + j, j * p + k, k * n + i, 1, 1, 1)
    
    def add_block_strassen(i_start, j_start, k_start):
        """Add Strassen terms for a 2x2x2 block starting at (i_start, j_start, k_start)."""
        # A block: rows i_start, i_start+1; cols j_start, j_start+1
        # B block: rows j_start, j_start+1; cols k_start, k_start+1
        # C block: rows k_start, k_start+1; cols i_start, i_start+1
        #
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B11-B01)
        # M5 = (A00+A10)(B01+B11)
        # M6 = (A11-A01)(B00+B10)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        i0, i1 = i_start, i_start + 1
        j0, j1 = j_start, j_start + 1
        k0, k1 = k_start, k_start + 1
        
        # M1 = (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
        add_term(i0 * m + j0, j0 * p + k0, k0 * n + i0, 1, 1, 1)   # M1 -> C[k0,i0]
        add_term(i0 * m + j0, j0 * p + k0, k1 * n + i1, 1, 1, 1)   # M1 -> C[k1,i1]
        add_term(i1 * m + j1, j0 * p + k0, k0 * n + i0, 1, 1, 1)   # M1 -> C[k0,i0]
        add_term(i1 * m + j1, j0 * p + k0, k1 * n + i1, 1, 1, 1)   # M1 -> C[k1,i1]
        add_term(i0 * m + j0, j1 * p + k1, k0 * n + i0, 1, 1, 1)   # M1 -> C[k0,i0]
        add_term(i0 * m + j0, j1 * p + k1, k1 * n + i1, 1, 1, 1)   # M1 -> C[k1,i1]
        add_term(i1 * m + j1, j1 * p + k1, k0 * n + i0, 1, 1, 1)   # M1 -> C[k0,i0]
        add_term(i1 * m + j1, j1 * p + k1, k1 * n + i1, 1, 1, 1)   # M1 -> C[k1,i1]
        
        # Wait, this is getting complicated. Let me think about it differently.
        # Each M is a product of a linear combination of A entries and a linear
        # combination of B entries. The result contributes to specific C entries.
        #
        # For the tensor decomposition, each term (a, b, c) means:
        # U has 1 at position corresponding to (i,j) = a
        # V has 1 at position corresponding to (j,k) = b  
        # W has 1 at position corresponding to (k,i) = c
        #
        # For Strassen, M1 = (A00 + A11)(B00 + B11)
        # This is: (A[i0,j0] * B[j0,k0] + A[i0,j0] * B[j1,k1] + A[i1,j1] * B[j0,k0] + A[i1,j1] * B[j1,k1])
        # Each of these 4 terms contributes to both C[k0,i0] and C[k1,i1].
        #
        # So M1 contributes 4 * 2 = 8 terms? No, that's not right either.
        #
        # Actually, in the tensor decomposition, we need:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = C[i,k]
        #
        # For M1 = (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1]), this equals:
        # A[i0,j0]*B[j0,k0] + A[i0,j0]*B[j1,k1] + A[i1,j1]*B[j0,k0] + A[i1,j1]*B[j1,k1]
        #
        # And this should equal C[k0,i0] + C[k1,i1] (with appropriate coefficients).
        #
        # In the tensor decomposition, we can represent this as 4 separate terms:
        # Term 1: U[i0*m+j0]=1, V[j0*p+k0]=1, W[k0*n+i0]=1, W[k1*n+i1]=1
        # But W can only have one entry per term... 
        #
        # Oh wait, I think I'm overcomplicating this. In the standard tensor
        # decomposition, each rank-1 term is:
        # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        #
        # For a single rank-1 term to contribute to C[i,k], we need:
        # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        #
        # This is a product of three scalars. For it to be non-zero, all three
        # must be non-zero. And the contribution to C[i,k] is this product.
        #
        # So for M1, which is a sum of 4 products, each product A[i',j']*B[j',k']
        # contributes to C[k',i'] for specific (i',k') pairs.
        #
        # Actually, I think the way to handle Strassen in this framework is:
        # Each "multiplication" M = (sum of A terms) * (sum of B terms)
        # can be expanded into individual A[i,j]*B[j,k] products.
        # But that defeats the purpose.
        #
        # Let me reconsider. The rank of a tensor decomposition is the minimum
        # number of rank-1 tensors that sum to the original tensor.
        #
        # For matrix multiplication, the tensor T[i,j,k] = 1 if we're computing
        # C[i,k] = sum_j A[i,j]*B[j,k].
        #
        # A rank-1 decomposition term is: sum_{i,j,k} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * e_i e_j e_k
        #
        # Wait, I think the convention is:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        #
        # No, that's not right. The tensor decomposition is:
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        #
        # And we need T[i,j,k] = 1 for all (i,j,k) since the matrix multiplication
        # tensor has all ones.
        #
        # Wait no. The matrix multiplication tensor is defined such that:
        # C = A * B means C[i,k] = sum_j A[i,j] B[j,k]
        #
        # The tensor T has T[i,j,k] = 1 for all valid (i,j,k).
        #
        # And a rank-r decomposition means:
        # T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c]
        #
        # where U is indexed by (i,j), V by (j,k), W by (k,i).
        #
        # For the schoolbook decomposition, each (i,j,k) triple gets its own term:
        # U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1 for c corresponding to (i,j,k).
        #
        # For Strassen, we group terms. For example, M1 = (A00+A11)(B00+B11)
        # This is one "multiplication" but it expands to 4 products.
        #
        # In the tensor decomposition, we can have a term where:
        # U[0,0,c]=1, U[1,1,c]=1 (two non-zero entries in U)
        # V[0,0,c]=1, V[1,1,c]=1 (two non-zero entries in V)
        # W[0,0,c]=1, W[1,1,c]=1 (two non-zero entries in W)
        #
        # Then the contribution is:
        # sum_{i,j,k} U[i,j,c]V[j,k,c]W[k,i,c] = 
        # U[0,0]V[0,0]W[0,0] + U[0,0]V[0,0]W[1,1] + ... 
        #
        # This doesn't quite work because the indices are coupled.
        
        # I think I need to just use the schoolbook decomposition for now
        # and see if I can find a better structure.
        pass
    
    # Let me just use schoolbook for now and verify it works.
    # Then I'll try to optimize.
    
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
