"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Use block decomposition: 5 = 2 + 2 + 1
    # Partition indices into blocks of sizes [2, 2, 1]
    block_sizes = [2, 2, 1]
    
    # For each pair of input blocks (i,j) and output block (k,l),
    # we need to compute the product of A[i,j] and B[j,k]
    # and add to C[i,k]
    
    # We'll use Strassen's algorithm for 2x2 block multiplications
    # and schoolbook for mixed sizes
    
    # First, let's build the tensor decomposition by considering
    # each (i,j,k) triple of block indices
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Helper: add a rank-1 term for multiplying block A[i,j] by B[j,k]
    # contributing to C[i,k]
    def add_term(A_idx, B_idx, C_idx, A_coeff, B_coeff, C_coeff):
        """
        A_idx = (block_row_A, block_col_A)
        B_idx = (block_row_B, block_col_B)  
        C_idx = (block_row_C, block_col_C)
        """
        br_a, bc_a = A_idx
        br_b, bc_b = B_idx
        br_c, bc_c = C_idx
        
        # Get the actual row ranges
        # A is n x m, partitioned by rows [2,2,1] and cols [2,2,1]
        # B is m x p, partitioned by rows [2,2,1] and cols [2,2,1]
        # C is p x n, partitioned by rows [2,2,1] and cols [2,2,1]
        
        # Compute row/col offsets
        def get_offsets(block_idx, sizes):
            start = 0
            for b in range(block_idx):
                start += sizes[b]
            return start, start + sizes[block_idx]
        
        # For A: rows are output rows of C, cols are input cols of B
        # Actually, let me think about this differently.
        
        # The tensor T_{i,j,k} = sum over middle index
        # For matrix mult: C_{i,k} = sum_j A_{i,j} B_{j,k}
        
        # In the block decomposition, for each combination of:
        # - A block row a_r, A block col a_c
        # - B block row b_r, B block col b_c  
        # - C block row c_r, C block col c_c
        # where a_c = b_r (the middle dimension must match)
        
        # We add terms for each such valid combination
        
        pass
    
    # Let me use a simpler direct approach:
    # Build the full tensor and then try to find a low-rank decomposition
    # by exploiting the block structure
    
    # Actually, let me just use the known result that rank(5x5) <= 100
    # and construct it using the formula from the literature.
    
    # The construction uses the following: for 5x5, we can achieve rank 100
    # by using a specific pattern.
    
    # Let me try a different approach: use the fact that
    # rank(n x n) <= n^2 * (n-1) + 1 is not always tight,
    # but for n=5, rank <= 100 is known.
    
    # I'll construct a rank-100 decomposition using the following method:
    # Use 100 rank-1 tensors that sum to the multiplication tensor.
    
    # One known construction: use the "binomial" approach
    # where we have terms for each (i, j, k) with specific coefficients.
    
    # Actually, the simplest valid improvement is to use the schoolbook
    # but verify it works, then try to optimize.
    
    # For now, return schoolbook to ensure validity, then we can iterate.
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
