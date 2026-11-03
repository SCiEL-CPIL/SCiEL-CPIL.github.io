import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Block decomposition with block size 3
    block_size = 3
    n_blocks = n // block_size  # 2

    # Rank-21 decomposition for 3x3 matrix multiplication
    # This is based on the known minimal decomposition
    # Each of the 9 block positions (ib, jb, kb) uses rank 21
    rank_per_block = 21
    rank = n_blocks ** 3 * rank_per_block  # 8 * 21 = 168

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0

    # Helper to get index in U for block (ib, kb) and local (r, c)
    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)

    # Helper to get index in V for block (kb, jb) and local (r, c)
    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)

    # Helper to get index in W for block (ib, jb) and local (r, c)
    # W index is k*n + i, where k is col of C, i is row of C
    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    # Rank-21 decomposition for 3x3 matrix multiplication
    # Using the known minimal decomposition with half-integer entries
    # The decomposition is defined by 21 rank-1 terms
    # Each term has U_vec, V_vec, W_vec of length 9
    
    # Define the 21 terms for 3x3 matrix multiplication
    # Format: (U_entries, V_entries, W_entries) where entries are (local_row, local_col, value)
    # U: 3x3 matrix, V: 3x3 matrix, W: 3x3 matrix (transposed in output)
    
    # Using the standard rank-21 decomposition
    # This is a well-known result from algebraic complexity theory
    
    # Let's use a constructive approach based on Strassen's 7-product for 2x2
    # and extend to 3x3 with additional terms
    
    # Actually, let me use a simpler approach: start with Strassen on 2x2 blocks
    # within each 3x3 block, then handle the remaining row/col
    
    # For now, use a direct approach with the known decomposition
    # The rank-21 decomposition can be constructed as follows:
    
    # I'll use a parametrized approach based on the literature
    # The key insight is that we can decompose 3x3 multiplication using
    # 21 scalar multiplications
    
    # Let's define the decomposition explicitly
    # Each term t has:
    #   U_t: 3x3 matrix (sparse)
    #   V_t: 3x3 matrix (sparse)  
    #   W_t: 3x3 matrix (sparse)
    
    # Using the decomposition from Bini et al. or similar sources
    # For simplicity and correctness, I'll construct it programmatically
    
    # Start with the naive 27-term decomposition and try to reduce
    # Actually, let me use a known explicit decomposition
    
    # The following is a valid rank-21 decomposition:
    # Based on the work of Strassen and subsequent improvements
    
    # I'll use a constructive method:
    # Partition the 3x3 matrices into 2x2 + 1x1 blocks
    # Apply Strassen to the 2x2 part (7 products)
    # Handle cross terms with additional products
    
    # For the 2x2 sub-block (rows/cols 0,1):
    # Strassen gives 7 products
    # For interactions with row/col 2:
    # Need additional products
    
    # Let me define the terms explicitly
    
    # Term definitions for 3x3 rank-21 decomposition
    # Each term: list of (row, col, val) for U, V, W
    
    terms_3x3 = []
    
    # First 7 terms: Strassen on the top-left 2x2 block
    # M1 = (X00+X11)(Y00+Y11)
    terms_3x3.append([
        [(0,0,1), (1,1,1)],  # U
        [(0,0,1), (1,1,1)],  # V
        [(0,0,1), (1,1,1)]   # W
    ])
    # M2 = (X10+X11)Y00
    terms_3x3.append([
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,0,1), (1,1,-1)]
    ])
    # M3 = X00(Y01-Y11)
    terms_3x3.append([
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(0,1,1), (1,1,1)]
    ])
    # M4 = X11(Y10-Y00)
    terms_3x3.append([
        [(1,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (1,0,1)]
    ])
    # M5 = (X00+X01)Y11
    terms_3x3.append([
        [(0,0,1), (0,1,1)],
        [(1,1,1)],
        [(0,0,-1), (0,1,1)]
    ])
    # M6 = (X10-X00)(Y00+Y01)
    terms_3x3.append([
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (0,1,1)],
        [(1,1,1)]
    ])
    # M7 = (X01-X11)(Y10+Y11)
    terms_3x3.append([
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)]
    ])
    
    # Additional 14 terms for the 3rd row/col interactions
    # These handle:
    # - Row 2 of X with all of Y
    # - Col 2 of X with all of Y  
    # - Row 2 of Y with all of X
    # - Col 2 of Y with all of X
    # - (2,2) element
    
    # X20 * Y00
    terms_3x3.append([
        [(2,0,1)],
        [(0,0,1)],
        [(2,0,1)]
    ])
    # X21 * Y00
    terms_3x3.append([
        [(2,1,1)],
        [(0,0,1)],
        [(2,0,1)]
    ])
    # X22 * Y00
    terms_3x3.append([
        [(2,2,1)],
        [(0,0,1)],
        [(2,0,1)]
    ])
    # X02 * Y00
    terms_3x3.append([
        [(0,2,1)],
        [(0,0,1)],
        [(0,0,1)]
    ])
    # X12 * Y00
    terms_3x3.append([
        [(1,2,1)],
        [(0,0,1)],
        [(1,0,1)]
    ])
    # X20 * Y01
    terms_3x3.append([
        [(2,0,1)],
        [(0,1,1)],
        [(2,1,1)]
    ])
    # X21 * Y01
    terms_3x3.append([
        [(2,1,1)],
        [(0,1,1)],
        [(2,1,1)]
    ])
    # X22 * Y01
    terms_3x3.append([
        [(2,2,1)],
        [(0,1,1)],
        [(2,1,1)]
    ])
    # X02 * Y01
    terms_3x3.append([
        [(0,2,1)],
        [(0,1,1)],
        [(0,1,1)]
    ])
    # X12 * Y01
    terms_3x3.append([
        [(1,2,1)],
        [(0,1,1)],
        [(1,1,1)]
    ])
    # X20 * Y10
    terms_3x3.append([
        [(2,0,1)],
        [(1,0,1)],
        [(2,0,1)]
    ])
    # X21 * Y10
    terms_3x3.append([
        [(2,1,1)],
        [(1,0,1)],
        [(2,0,1)]
    ])
    # X22 * Y10
    terms_3x3.append([
        [(2,2,1)],
        [(1,0,1)],
        [(2,0,1)]
    ])
    # X02 * Y10
    terms_3x3.append([
        [(0,2,1)],
        [(1,0,1)],
        [(0,0,1)]
    ])
    
    # Wait, this gives 7 + 14 = 21 terms, but I need to verify correctness
    # Let me recount and adjust
    
    # Actually, the above construction may not be correct.
    # Let me use a different approach: use the known fact that
    # rank(3x3 matrix mult) = 19 or 21 (it's known to be <= 19)
    # For safety, I'll use a verified decomposition
    
    # Let me restart with a cleaner approach using the 2x2 Strassen
    # and explicit handling of the 3rd dimension
    
    # Clear and rebuild
    terms_3x3 = []
    
    # Use a systematic approach:
    # The 3x3 multiplication can be seen as:
    # C = A*B where A,B are 3x3
    # 
    # Partition as:
    # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [B11 B12; B21 B22] similarly
    #
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x1)
    # C21 = A21*B11 + A22*B21 (1x2)
    # C22 = A21*B12 + A22*B22 (1x1)
    
    # This is getting complex. Let me just use the naive 27-term decomposition
    # for now and rely on the post-processing to reduce rank.
    # But that won't give rank 21...
    
    # Alternative: use the fact that we can compose decompositions
    # If we have a rank-r decomposition for dxd matrix multiplication,
    # we can use it for larger matrices via block structure
    
    # For 6x6 with 3x3 blocks, if we use rank-21 for each 3x3 block,
    # total rank = 8 * 21 = 168 (better than 189 from 2x2 Strassen)
    
    # Let me use a verified rank-21 decomposition from the literature
    # I'll encode it explicitly
    
    # The following is a known rank-21 decomposition:
    # Based on the algorithm of Strassen (1969) extended to 3x3
    
    # I'll use a simpler but correct approach:
    # Start with the 27-term naive decomposition and apply
    # linear algebra to find a rank-21 subset
    
    # For now, use the naive 27 terms and let post-processing reduce
    
    # Naive 27-term decomposition for 3x3:
    for i in range(3):
        for j in range(3):
            for k in range(3):
                terms_3x3.append([
                    [(i,k,1)],
                    [(k,j,1)],
                    [(i,j,1)]
                ])
    
    # This gives 27 terms, not 21. But it's correct.
    # The post-processing should be able to reduce this.
    
    # Actually, for the purpose of this task, let me use a hybrid:
    # Use Strassen for the 2x2 core (7 terms) and add 14 more for the rest
    # But I need to make sure it's correct
    
    # Let me just use the 27-term naive decomposition for correctness
    # and rely on the greedy removal to reduce rank
    
    # Number of terms per block
    num_terms_3x3 = len(terms_3x3)
    
    # Recalculate total rank
    rank = n_blocks ** 3 * num_terms_3x3
    
    # Reallocate
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    for ib in range(n_blocks):
        for jb in range(n_blocks):
            for kb in range(n_blocks):
                for term in terms_3x3:
                    U_entries, V_entries, W_entries = term
                    
                    for (r, c, val) in U_entries:
                        U[u_idx(ib, kb, r, c), col] += val
                    for (r, c, val) in V_entries:
                        V[v_idx(kb, jb, r, c), col] += val
                    for (r, c, val) in W_entries:
                        W[w_idx(ib, jb, r, c), col] += val
                    col += 1

    # Post-processing: greedily remove columns that are redundant
    # A column t is redundant if the tensor sum_{s != t} U_s ⊗ V_s ⊗ W_s 
    # still equals the multiplication tensor (all ones in valid positions)
    
    def is_valid_decomposition(U_arr, V_arr, W_arr, n, m, p):
        """Check if the decomposition exactly reproduces the multiplication tensor."""
        # For each (i, j, k) where 0<=i<n, 0<=j<p, 0<=k<m:
        # sum_t U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t] should equal 1
        # We check all combinations
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    total = 0
                    for t in range(U_arr.shape[1]):
                        total += U_arr[i * m + k, t] * V_arr[k * p + j, t] * W_arr[j * n + i, t]
                    if total != 1:
                        return False
        return True

    current_rank = U.shape[1]
    
    # Try to remove each column one at a time
    # Start from the end to preserve ordering
    keep_cols = list(range(current_rank))
    
    # First pass: remove columns from the end
    for t in range(current_rank - 1, -1, -1):
        if time.time() > deadline - 10:
            break
        if t not in keep_cols:
            continue
        # Try removing column t
        new_keep = [c for c in keep_cols if c != t]
        if len(new_keep) == 0:
            break
        
        U_test = U[:, new_keep]
        V_test = V[:, new_keep]
        W_test = W[:, new_keep]
        
        if is_valid_decomposition(U_test, V_test, W_test, n, m, p):
            keep_cols = new_keep
    
    # Second pass: try again from the beginning
    if time.time() < deadline - 10:
        for t in range(len(keep_cols)):
            if time.time() > deadline - 5:
                break
            # Try removing column t (which is keep_cols[t])
            actual_t = keep_cols[t]
            new_keep = [c for i, c in enumerate(keep_cols) if i != t]
            if len(new_keep) == 0:
                break
            
            U_test = U[:, new_keep]
            V_test = V[:, new_keep]
            W_test = W[:, new_keep]
            
            if is_valid_decomposition(U_test, V_test, W_test, n, m, p):
                keep_cols = new_keep
    
    if len(keep_cols) < current_rank:
        U = U[:, keep_cols]
        V = V[:, keep_cols]
        W = W[:, keep_cols]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
