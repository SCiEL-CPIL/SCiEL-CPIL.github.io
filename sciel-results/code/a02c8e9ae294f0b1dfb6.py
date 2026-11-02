<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (16, rank)  
    # W: (p*n, rank) = (12, rank)
    
    def schoolbook():
        rank = n * m * p  # 48
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition is correct."""
        rank = U.shape[1]
        # Build the tensor T_{(i,k),(i',j'),(j',k')} = delta_{i,i'}*delta_{k,k'}
        # Reconstruction: for each output (i,k), sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r]*A[i',j']) * (sum_{j',k'} V[j'*p+k', r]*B[j',k'])
        # This should equal sum_j A[i,j]*B[j,k]
        
        # Test with random matrices
        rng = np.random.RandomState(42)
        for _ in range(3):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_expected = A @ B
            
            # Compute via decomposition
            # L_A[r] = sum_{i',j'} U[i',j',r] * A[i',j']
            L_A = np.zeros(rank)
            for r in range(rank):
                L_A[r] = np.sum(U[:, r] * A.flatten())
            
            L_B = np.zeros(rank)
            for r in range(rank):
                L_B[r] = np.sum(V[:, r] * B.flatten())
            
            C_recon = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    C_recon[i, k] = np.sum(W[k*n + i, :] * L_A * L_B)
            
            if np.max(np.abs(C_recon - C_expected)) > tol:
                return False
        return True
    
    def try_reduce_rank(U, V, W, target_rank, time_limit):
        """Try to reduce rank to target_rank by zeroing out columns."""
        start_time = time.time()
        rank = U.shape[1]
        if target_rank >= rank:
            return U, V, W, rank
        
        # Try all subsets of size target_rank from the current rank
        # This is C(48, 24) which is huge, so we need a smarter approach
        # Instead, try to find a subset of columns that spans the tensor
        
        # Greedy: iteratively remove columns if the decomposition still works
        U_cur = U.copy()
        V_cur = V.copy()
        W_cur = W.copy()
        current_rank = rank
        
        # Try removing columns one at a time
        removed = []
        for col in range(rank):
            if time.time() - start_time > time_limit:
                break
            if col in removed:
                continue
            # Try removing this column
            mask = [c for c in range(rank) if c != col and c not in removed]
            if len(mask) < target_rank:
                break
            
            U_test = U_cur[:, mask]
            V_test = V_cur[:, mask]
            W_test = W_cur[:, mask]
            
            if verify(U_test, V_test, W_test):
                removed.append(col)
                current_rank = len(mask)
        
        if current_rank < rank:
            mask = [c for c in range(rank) if c not in removed]
            return U_cur[:, mask], V_cur[:, mask], W_cur[:, mask], current_rank
        
        return U, V, W, rank
    
    # Start with schoolbook
    U, V, W = schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    start_time = time.time()
    time_budget = 120  # seconds, with safety margin
    
    # Try to reduce rank
    # First, try greedy removal from full rank
    U_reduced, V_reduced, W_reduced, rank_reduced = try_reduce_rank(U, V, W, 1, time_budget * 0.3)
    
    if rank_reduced < best_rank:
        best_rank = rank_reduced
        best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
    
    # Try different random initial orderings to find better reductions
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 5:
        rng = np.random.RandomState(123)
        for trial in range(5):
            if time.time() - start_time > time_budget:
                break
            # Randomly permute the columns
            perm = rng.permutation(best_rank)
            U_perm = best_U[:, perm]
            V_perm = best_V[:, perm]
            W_perm = best_W[:, perm]
            
            U_try, V_try, W_try, rank_try = try_reduce_rank(U_perm, V_perm, W_perm, 1, remaining_time / 6)
            
            if rank_try < best_rank:
                best_rank = rank_try
                best_U, best_V, best_W = U_try, V_try, W_try
    
    # Convert to lists for output
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (16, rank)  
    # W: (p*n, rank) = (12, rank)
    
    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition is correct."""
        rank = U.shape[1]
        rng = np.random.RandomState(42)
        for _ in range(3):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_expected = A @ B
            
            L_A = np.zeros(rank)
            for r in range(rank):
                L_A[r] = np.sum(U[:, r] * A.flatten())
            
            L_B = np.zeros(rank)
            for r in range(rank):
                L_B[r] = np.sum(V[:, r] * B.flatten())
            
            C_recon = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    C_recon[i, k] = np.sum(W[k*n + i, :] * L_A * L_B)
            
            if np.max(np.abs(C_recon - C_expected)) > tol:
                return False
        return True
    
    def strassen_2x2():
        """Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
        
        Returns U, V, W each of shape (4, 7) for 2x2 matrices.
        Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen's 7 products:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = -1
        W[1, 0] = 1; W[2, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[3, 6] = 1
        
        return U, V, W
    
    def block_decomposition():
        """Construct decomposition using Strassen for 2x2 blocks and trivial for remaining.
        
        Strategy: Split the 4x4 multiplication into 2x2 blocks.
        Use Strassen (rank 7) for the 2x2 block multiplications.
        
        For 3x4 times 4x4:
        - Split A into 3 rows, B into 4 columns
        - Group columns of B into pairs: (0,1) and (2,3)
        - For each pair, compute 3x2 * 2x2 using block structure
        
        Actually, let's use a cleaner approach:
        - Split the 4x4 matrix B into 2x2 blocks
        - For each output row i (0,1,2) and each 2x2 block of B:
          - Use Strassen if the 2x2 block multiplication can be shared
          - Use trivial otherwise
        
        Better approach: Use the tiling strategy from the inspiration.
        Split into 2x2 blocks. The 3x4 * 4x4 can be viewed as:
        - 3 rows of A, each with 4 elements
        - Split A into 3x2 blocks (two blocks)
        - Split B into 2x2 blocks (four blocks)
        
        For block (i, block_j) where i is row group and block_j is 2x2 block of B:
        - We need to compute (3x2) * (2x2) -> (3x2)
        - This can be done with rank 11 (Strassen for 2x2 + trivial for extra row)
        
        Total: 2 blocks of A * 4 blocks of B = 8 blocks, each rank 11 = 88? No, that's worse.
        
        Let me think differently. The 4x4 multiplication has rank 19 (known result).
        For 3x4 * 4x4, we can use:
        - For each of the 3 rows of A, multiply by B (4x4)
        - Each row is a 1x4 vector, result is 1x4
        - But we can share computation across rows
        
        Actually, the cleanest approach: 
        - Use the known rank-19 algorithm for 4x4x4
        - For 3x4 * 4x4, we only need 3 rows of the 4x4 result
        - So we can potentially reduce the rank
        
        Let me try a direct construction:
        - Split into 2x2 blocks
        - A is 3x4: split into rows [0,1] (2 rows) and [2] (1 row)
        - B is 4x4: split into 2x2 blocks
        
        For the 2x2 * 2x2 part (rows 0,1 of A):
          Use Strassen with rank 7 for each 2x2 block pair
          But we have 2x2 blocks in both dimensions...
        
        Let me just implement the tiling approach that gave rank 44:
        - 4 blocks of 3x2x2, each with rank 11
        - Total rank: 4 * 11 = 44
        """
        # Tiling approach: split into 4 blocks of size 3x2x2
        # Each block: (3x2) * (2x2) -> (3x2)
        # Use Strassen for the 2x2x2 part + trivial for the extra row
        
        U_blocks = []
        V_blocks = []
        W_blocks = []
        
        for block in range(4):
            # Each block handles columns 2*block and 2*block+1 of B
            # And rows 0,1,2 of A
            # Input: A[:, 2*block:2*block+2] (3x2) * B[2*block:2*block+2, :] (2x2) -> (3x2)
            
            # For this block, we need to compute:
            # C[i, 2*block] = A[i, 2*block]*B[2*block, 2*block] + A[i, 2*block+1]*B[2*block+1, 2*block]
            # C[i, 2*block+1] = A[i, 2*block]*B[2*block, 2*block+1] + A[i, 2*block+1]*B[2*block+1, 2*block+1]
            
            # Use Strassen for the 2x2 part (rows 0,1 of A)
            # And trivial for row 2 of A
            
            U_block = np.zeros((6, 11))  # 3 rows * 2 cols = 6 inputs
            V_block = np.zeros((4, 11))   # 2 rows * 2 cols = 4 inputs
            W_block = np.zeros((4, 11))   # 2 outputs * 2 rows = 4 outputs
            
            # Strassen for rows 0,1 (indices 0,1 in the 3x2 block)
            sU, sV, sW = strassen_2x2()
            
            # Map Strassen indices to our block
            # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
            # B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
            # C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
            
            for r in range(7):
                U_block[0:4, r] = sU[:, r]
                V_block[:, r] = sV[:, r]
                W_block[0:4, r] = sW[:, r]
            
            # Trivial for row 2 (index 2 in the 3x2 block, which is indices 4,5 in U_block)
            # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
            # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
            
            # Product 7: A[2,0] * B[0,0]
            U_block[4, 7] = 1
            V_block[0, 7] = 1
            W_block[2, 7] = 1  # C[2,0]
            
            # Product 8: A[2,0] * B[0,1]
            U_block[4, 8] = 1
            V_block[1, 8] = 1
            W_block[3, 8] = 1  # C[2,1]
            
            # Product 9: A[2,1] * B[1,0]
            U_block[5, 9] = 1
            V_block[2, 9] = 1
            W_block[2, 9] = 1  # C[2,0]
            
            # Product 10: A[2,1] * B[1,1]
            U_block[5, 10] = 1
            V_block[3, 10] = 1
            W_block[3, 10] = 1  # C[2,1]
            
            U_blocks.append(U_block)
            V_blocks.append(V_block)
            W_blocks.append(W_block)
        
        # Combine all blocks
        U = np.hstack(U_blocks)  # (6*4, 11*4) = (24, 44) -- wait, that's wrong
        # Actually, each block has its own U, V, W
        # U is (n*m, rank) = (12, 44)
        # V is (m*p, rank) = (16, 44)
        # W is (p*n, rank) = (12, 44)
        
        U_full = np.zeros((12, 44))
        V_full = np.zeros((16, 44))
        W_full = np.zeros((12, 44))
        
        for block in range(4):
            col_offset = block * 11
            
            # U: A[i, j] where i in 0..2, j in 2*block..2*block+1
            # In U_full, row index is i*m + j = i*4 + j
            # In U_block, row index is i*2 + (j - 2*block)
            for i in range(3):
                for j in range(2):
                    U_full[i*4 + (2*block + j), col_offset:col_offset+11] = U_block[i*2 + j, :]
            
            # V: B[j, k] where j in 2*block..2*block+1, k in 0..3
            # In V_full, row index is j*p + k = j*4 + k
            # In V_block, row index is (j - 2*block)*2 + k
            for j in range(2):
                for k in range(4):
                    V_full[(2*block + j)*4 + k, col_offset:col_offset+11] = V_block[(j)*2 + k, :]
            
            # W: C[i, k] where i in 0..2, k in 2*block..2*block+1
            # In W_full, row index is k*n + i = k*3 + i
            # In W_block, row index is (k - 2*block)*2 + i
            for i in range(3):
                for k in range(2):
                    W_full[(2*block + k)*3 + i, col_offset:col_offset+11] = W_block[(k)*2 + i, :]
        
        return U_full, V_full, W_full
    
    # Construct the block decomposition
    U, V, W = block_decomposition()
    
    # Verify
    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    
    # Convert to lists for output
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
