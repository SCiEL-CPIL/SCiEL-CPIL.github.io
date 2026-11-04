<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Use Strassen-based block decomposition (rank 33) as baseline,
    then search for lower ranks using random and structured methods.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, T)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def strassen_block_decomposition():
        """
        Return rank-33 decomposition: 3 blocks of (Strassen 2x2x2 + Standard 2x1x2).
        Each block handles a 2x2 sub-block of the 3x6 B matrix plus the remaining column.
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        # Strassen 2x2x2 factors
        S_U = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        S_V = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
        S_W = [
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]
        
        # Standard 2x1x2 factors
        Std_U = [
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ]
        Std_V = [
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ]
        Std_W = [
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ]
        
        col = 0
        for b in range(3):
            # Strassen part: handles columns 2b, 2b+1 of B (and corresponding C)
            for r in range(7):
                # Map U (2x2) to global U (rows for i=0,1 and j=0,1)
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r][i][j]
                
                # Map V (2x2) to global V (rows for j=0,1 and k=2b, 2b+1)
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r][j][kp]
                
                # Map W (2x2) to global W (rows for k=2b, 2b+1 and i=0,1)
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r][kp][i]
                
                col += 1
            
            # Standard part: handles column j=2 of A/B, columns 2b, 2b+1 of B
            for r in range(4):
                # Map U (2x1) to global U (rows for i=0,1 and j=2)
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r][i][0]
                
                # Map V (1x2) to global V (row for j=2 and k=2b, 2b+1)
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r][0][kp]
                
                # Map W (2x2) to global W (rows for k=2b, 2b+1 and i=0,1)
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r][kp][i]
                
                col += 1
        
        return U, V, W

    # Start with the known rank-33 decomposition
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33
    
    # Verify the baseline
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook if Strassen construction is wrong
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    # Compute lower bounds from matricizations
    T1 = T.reshape(dim_a, dim_b * dim_c)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = T.reshape(dim_b, dim_a * dim_c)
    rank2 = np.linalg.matrix_rank(T2)
    T3 = T.reshape(dim_c, dim_a * dim_b)
    rank3 = np.linalg.matrix_rank(T3)
    lower_bound = max(rank1, rank2, rank3)

    # Search for lower ranks
    found_improvement = False
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try random search for this rank with various strategies
        max_attempts = 3000
        attempts = 0
        while not found_improvement and attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            
            # Strategy 1: Random integer vectors in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(dim_a, r))
            V = np.random.randint(-1, 2, size=(dim_b, r))
            W = np.random.randint(-1, 2, size=(dim_c, r))
            
            if verify_decomposition(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                found_improvement = True
                break
            
            # Strategy 2: Sparse random (mostly zeros with occasional non-zero)
            if attempts % 3 == 0:
                U = np.zeros((dim_a, r), dtype=np.int64)
                V = np.zeros((dim_b, r), dtype=np.int64)
                W = np.zeros((dim_c, r), dtype=np.int64)
                for c in range(r):
                    # Pick 1-3 non-zero entries per column
                    for _ in range(np.random.randint(1, 4)):
                        U[np.random.randint(dim_a), c] = np.random.choice([-1, 1])
                    for _ in range(np.random.randint(1, 4)):
                        V[np.random.randint(dim_b), c] = np.random.choice([-1, 1])
                    for _ in range(np.random.randint(1, 4)):
                        W[np.random.randint(dim_c), c] = np.random.choice([-1, 1])
                
                if verify_decomposition(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    found_improvement = True
                    break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Use structured algebraic search to find low-rank decompositions.
    """
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s margin for safety

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition_fast(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T using matrix operations."""
        rank = U.shape[1]
        # For each c-index, check if T[:, :, c] = sum_r U[:,r] * V[:,r] * W[c,r]
        # Reshape: for fixed c, T[:, :, c] is (dim_a, dim_b)
        # U[:,r] * V[:,r] is outer product (dim_a, dim_b)
        # So T[:, :, c] = sum_r W[c,r] * outer(U[:,r], V[:,r])
        # This is: T[:,:,c] = (U * W[c,:]) @ V.T  -- no, let me think again
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed c: T[:,:,c] = sum_r U[:,r] * W[c,r] * V[:,r].T
        # = (U * W[c,:]) @ V.T  where * is element-wise scaling of columns
        # Actually: T[:,:,c] = sum_r W[c,r] * np.outer(U[:,r], V[:,r])
        # = (U * W[c,:].reshape(1,-1)) @ V.T  -- this works!
        
        # Vectorized check:
        # result[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # = einsum('ar,br,cr->abc', U, V, W)
        
        # Check all c at once:
        # For each c, T[:,:,c] should equal sum_r W[c,r] * outer(U[:,r], V[:,r])
        # Let's compute: R = np.einsum('ar,cr->ac', U, W)  # (dim_a, rank) * (dim_c, rank) -> no
        # Actually: result[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # = sum_r (U[a,r]*W[c,r]) * V[b,r]
        # Let M = U * W  element-wise? No.
        # M[a,c,r] = U[a,r]*W[c,r], then result[a,b,c] = sum_r M[a,c,r]*V[b,r]
        # This is getting complex. Let me just do it directly but efficiently.
        
        # Efficient approach: 
        # result = np.einsum('ar,br,cr->abc', U, V, W)
        # But this creates a large intermediate. Let's check per c.
        
        for c in range(dim_c):
            # T[:,:,c] vs sum_r W[c,r] * outer(U[:,r], V[:,r])
            expected = T[:, :, c]
            actual = np.zeros((dim_a, dim_b), dtype=np.int64)
            for r in range(rank):
                if W[c, r] != 0:
                    actual += W[c, r] * np.outer(U[:, r], V[:, r])
            if not np.array_equal(expected, actual):
                return False
        return True

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def strassen_block_decomposition():
        """
        Return rank-33 decomposition: 3 blocks of (Strassen 2x2x2 + Standard 2x1x2).
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        S_U = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        S_V = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
        S_W = [
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]
        
        Std_U = [
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ]
        Std_V = [
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ]
        Std_W = [
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ]
        
        col = 0
        for b in range(3):
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r][i][j]
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r][j][kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r][kp][i]
                col += 1
            
            for r in range(4):
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r][i][0]
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r][0][kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r][kp][i]
                col += 1
        
        return U, V, W

    # Start with the known rank-33 decomposition
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33
    
    if not verify_decomposition_fast(best_U, best_V, best_W):
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    # Compute lower bounds from matricizations
    T1 = T.reshape(dim_a, dim_b * dim_c)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = T.reshape(dim_b, dim_a * dim_c)
    rank2 = np.linalg.matrix_rank(T2)
    T3 = T.reshape(dim_c, dim_a * dim_b)
    rank3 = np.linalg.matrix_rank(T3)
    lower_bound = max(rank1, rank2, rank3)

    # Try to find lower rank decompositions using structured approaches
    # Key insight: try to merge terms in the Strassen decomposition
    
    # Approach: Try to find a rank-r decomposition by solving a system
    # For each candidate rank r, we need to find U, V, W such that
    # sum_r U[:,r] * V[:,r] * W[:,r] = T
    
    # Try ranks from lower_bound to best_rank - 1
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 15:
            break
        
        # Strategy: Greedy construction
        # Start with zero matrices and try to satisfy each tensor element
        found = False
        
        # Try multiple random initializations with structured patterns
        num_trials = 200
        for trial in range(num_trials):
            if time.time() > deadline - 15:
                break
            
            # Generate structured random decomposition
            # Use small integers: -2, -1, 0, 1, 2
            U = np.random.randint(-2, 3, size=(dim_a, r)).astype(np.int64)
            V = np.random.randint(-2, 3, size=(dim_b, r)).astype(np.int64)
            W = np.random.randint(-2, 3, size=(dim_c, r)).astype(np.int64)
            
            # Ensure each column has at least one non-zero
            for c in range(r):
                if np.all(U[:, c] == 0):
                    U[np.random.randint(dim_a), c] = np.random.choice([-1, 1])
                if np.all(V[:, c] == 0):
                    V[np.random.randint(dim_b), c] = np.random.choice([-1, 1])
                if np.all(W[:, c] == 0):
                    W[np.random.randint(dim_c), c] = np.random.choice([-1, 1])
            
            if verify_decomposition_fast(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                found = True
                break
        
        if found:
            # Continue searching for even lower ranks
            continue
    
    # Final verification
    if not verify_decomposition_fast(best_U, best_V, best_W):
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I realize the random search approach is still fundamentally flawed - the probability of randomly hitting an exact decomposition is essentially zero. Let me think more carefully.

The key insight is that for matrix multiplication tensors, there are known algebraic structures. For 2×3 × 3×6, we can think of this as:
- A is 2×3, B is 3×6, C = A×B is 2×6

The standard approach would give rank 36. Strassen on 2×2 blocks gives 7 per 2×2×2 block. If we split B into three 3×2 blocks, each block multiplication is 2×3 × 3×2. 

For 2×3 × 3×2: We can write this as (2×2 × 2×2) + (2×1 × 1×2) terms. The 2×2×2 part has rank 7 (Strassen), and the 2×1×1×2 part... let me think.

Actually, 2×3 × 3×2 can be decomposed by splitting the middle dimension:
- (2×2 × 2×2) for the first 2 columns of B, plus (2×1 × 1×2) for the last column
- Rank: 7 + 4 = 11? But that's not quite right because the W factors need to align.

Let me reconsider. The current rank-33 approach uses 3 blocks of (7+4)=11, giving 33. 

Can we do better? What if we use a different block structure? 

For 2×3 × 3×6, one approach: split the 3 in the middle into 2+1.
- Part 1: 2×2 × 2×6 (rank for 2×2×6?)
- Part 2: 2×1 × 1×6 (rank 6)

For 2×2×6: Split the 6 into three 2s: 3 × (2×2×2) = 3×7 = 21
Plus 2×1×1×6: rank 6
Total: 21+6 = 27? But the W factors need to be consistent.

Actually this isn't quite right because the decomposition needs to be a valid bilinear decomposition of the whole tensor, not just a sum of independent sub-decompositions unless they act on disjoint supports.

Let me try a completely different approach: use the fact that we can represent the tensor and try to find a low-rank decomposition using algebraic methods.

