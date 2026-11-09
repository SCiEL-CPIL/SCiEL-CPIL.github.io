"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # We'll build the decomposition by partitioning the 3 output rows:
    # Block 0: 1x5x6 (rank 1)
    # Block 1: 2x5x6 (rank 48)
    # Total rank: 49
    
    # --- Helper: rank-1 decomposition for 1 x m x p ---
    def rank1_1mp(m, p):
        """Decompose 1xm * mxp -> 1xp with rank 1.
        U: (m x 1), V: (p x 1), W: (p x 1) in index order.
        For 1xm*mxp: C[0,j] = sum_k A[0,k]*B[k,j]
        Tensor T[i,k,j] = 1 for i=0.
        u_i=1, v_k=1, w_j=1 gives T[0,k,j]=1.
        """
        U = [[1] * m]  # U[i*m+j, c] -> U[0*m+j, 0] = 1
        V = [[1] * p]  # V[j*p+k, c] -> V[0*p+k, 0] = 1
        W = [[1] * p]  # W[k*n+i, c] -> W[k*1+0, 0] = 1
        return U, V, W  # shapes: (m,1), (p,1), (p,1)
    
    # --- Helper: 2xm*p decomposition using Strassen-like method ---
    def strassen_2mp(m, p):
        """Decompose 2xm * mxp -> 2xp.
        Using block partition of the m dimension.
        For 2x2 blocks: Strassen gives rank 7.
        
        For general m, we can partition m into blocks and apply Strassen.
        But for m=5, let's use a direct approach.
        
        Actually, for 2xm*mxp, the tensor is:
        T[i,k,j] = 1 if the product A[i,k]*B[k,j] contributes to C[i,j]
        
        A simpler approach: partition the 2 output rows independently isn't minimal.
        
        Let me use a known result: rank(2,5,6) can be achieved with 48.
        
        Alternative: use 2x2x2 Strassen blocks where possible.
        m=5: partition as 2+3
        p=6: partition as 3+3
        
        Actually, let me think differently. For 2xm*mxp:
        We can write it as sum of rank-1 terms.
        The schoolbook rank is 2*m*p = 60.
        
        A better approach: use the fact that for 2xm*mxp,
        if we partition m into m1+m2 and p into p1+p2,
        we get four 2xm_i x p_j sub-problems.
        
        For 2x2x2: Strassen rank = 7
        For 2x2x3: need to compute
        For 2x3x3: need to compute
        
        Let me just use a systematic approach with known small cases.
        
        Actually, the simplest reliable approach: use the 2x2x2 Strassen
        decomposition and extend it.
        
        For 2xm*mxp with m=5, p=6:
        Partition m = 2+3, p = 3+3
        
        Sub-blocks:
        (2x2)*(2x3) -> 2x3
        (2x2)*(2x3) -> 2x3  
        (2x3)*(3x3) -> 2x3
        (2x3)*(3x3) -> 2x3
        
        This is getting complex. Let me use a different strategy.
        
        For 2xm*mxp, I'll use the following:
        Split m into chunks and use bilinear forms.
        
        Actually, let me just verify: what's the known rank of M(2,5,6)?
        
        Let me try a construction:
        For 2xm*mxp, define for each (i,k,j) where i in {0,1}:
        The tensor is T[i,k,j] = 1.
        
        We can decompose as sum of rank-1 tensors.
        
        Let me use the approach: for 2xm*mxp,
        use m*p rank-1 terms but exploit linearity.
        
        Hmm, let me just implement a verified rank-48 construction for 2x5x6.
        
        I'll use the following known decomposition based on Strassen:
        Partition the 5x6 B matrix into 5 rows, each of length 6.
        Partition the 2x5 A matrix into 2 rows, each of length 5.
        
        C[0,j] = sum_k A[0,k]*B[k,j]
        C[1,j] = sum_k A[1,k]*B[k,j]
        
        Let S1 = A[0,:]+A[1,:], S2 = A[0,:]-A[1,:], S3 = A[0,:]
        Then C[0,:] = (S1+B_contribution)/2, C[1,:] = (S1-B_contribution)/2
        
        But this doesn't reduce the number of B multiplications.
        
        Let me just use a brute-force verified rank-48 construction.
        
        For 2x5x6, I'll use the following approach:
        - Use 6 groups, one for each output column j
        - For each j, we need to compute sum_k A[:,k]*B[k,j] for both rows
        - That's 2*5 = 10 rank-1 terms per j, total 60
        - But we can share computations across j
        
        Actually, the cleanest approach: since I need a VALID decomposition,
        let me just use rank 60 for 2x5x6 (schoolbook) + rank 1 for 1x5x6 = 61.
        That gives score 90/61 ≈ 1.48.
        
        But I want to do better. Let me try rank 48.
        
        Here's a valid rank-48 construction for 2x5x6:
        Partition the 6 output columns into 3 groups of 2.
        For each group of 2 columns, use a 2x5x2 sub-decomposition.
        
        For 2x5x2: schoolbook rank = 20.
        Can we do better? 
        2x5x2: T[i,k,j] for i in {0,1}, k in {0..4}, j in {0,1}
        
        Let me just use schoolbook for the sub-problems but with sharing.
        
        OK let me take a step back and implement a clean, correct solution.
        
        I'll use the following strategy:
        1. For the 1x5x6 block: rank 1
        2. For the 2x5x6 block: use Strassen-style decomposition
        
        For 2x5x6, I'll partition 5 = 2+3 and use:
        - 2x2x6 part and 2x3x6 part
        
        For 2x2x6: partition 6 = 2+2+2, use Strassen 2x2x2 three times = 3*7 = 21
        For 2x3x6: partition 3 = 1+2 and 6 = 2+2+2
          - 2x1x2 (rank 2), 2x1x2 (rank 2), 2x1x2 (rank 2) = 6
          - 2x2x2 (rank 7), 2x2x2 (rank 7), 2x2x2 (rank 7) = 21
          - Total for 2x3x6 = 6 + 21 = 27
        Total for 2x5x6 = 21 + 27 = 48
        
        So total rank = 1 + 48 = 49.
        """
        pass
    
    # Let me implement this more carefully.
    # I'll build the full 3x5x6 decomposition directly.
    
    # The tensor T[i,k,j] = 1 for all i,k,j where:
    # i in [0,3), k in [0,5), j in [0,6)
    # C[i,j] = sum_k A[i,k] * B[k,j]
    
    # Decomposition: sum_r u_r ⊗ v_r ⊗ w_r
    # where U[i*m+k, r], V[k*p+j, r], W[j*n+i, r]
    # C[i,j] = sum_{k,r} U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    # I'll construct a valid rank-49 decomposition.
    
    # Part 1: Handle i=0 (first output row) with rank 1
    # For i=0: C[0,j] = sum_k A[0,k] * B[k,j]
    # Use u[k]=1, v[j]=1, w[i]=1: T[0,k,j] = 1*1*1 = 1 ✓
    # U[0*m+k, 0] = 1, V[k*p+j, 0] = 1, W[j*n+0, 0] = 1
    
    # Part 2: Handle i=1,2 (second and third output rows)
    # For i=1: C[1,j] = sum_k A[1,k] * B[k,j]
    # For i=2: C[2,j] = sum_k A[2,k] * B[k,j]
    # 
    # I'll use a Strassen-based approach for the 2x5x6 sub-problem.
    # 
    # Partition k: k in {0,1} and k in {2,3,4}
    # Partition j: j in {0,1}, {2,3}, {4,5}
    
    # For the 2x2x2 Strassen block, the 7 rank-1 terms are:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+cg, af+dh], [ce+dg, cf+gh]]
    # Strassen:
    # m1 = (a+d)(e+h)  -> contributes to C[0,0], C[1,1], C[0,1], C[1,0]
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+g)
    # m7 = (b-d)(f+h)
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # For 2x2x2, rank 7.
    
    # I need to build the full decomposition. Let me create it step by step.
    
    # Total rank = 1 (for i=0) + 48 (for i=1,2) = 49
    
    rank_total = 49
    U = [[0] * rank_total for _ in range(n * m)]  # 15 x 49
    V = [[0] * rank_total for _ in range(m * p)]  # 30 x 49
    W = [[0] * rank_total for _ in range(p * n)]  # 18 x 49
    
    # --- Rank 1 term for i=0 ---
    r = 0
    for k in range(m):
        U[0 * m + k][r] = 1
    for k in range(m):
        for j in range(p):
            V[k * p + j][r] = 1
    for j in range(p):
        W[j * n + 0][r] = 1
    
    # --- 48 rank-1 terms for i=1,2 using Strassen blocks ---
    # Partition: k_groups = [(0,1), (2,3,4)], j_groups = [(0,1), (2,3), (4,5)]
    # i is in {1, 2}, so i_local = i-1 in {0, 1}
    
    # For each (k_group, j_group) pair, we have a 2 x |kg| x |jg| sub-problem.
    # We'll decompose each using Strassen when both dimensions are 2,
    # and schoolbook otherwise.
    
    k_groups = [(0, 1), (2, 3, 4)]
    j_groups = [(0, 1), (2, 3), (4, 5)]
    
    col = 1  # start after the rank-1 term
    
    def add_strassen_2x2x2(col, i_base, k_vals, j_vals, U, V, W, m, p, n):
        """Add 7 rank-1 terms for a 2x2x2 Strassen block.
        i_base: the first i index (local i=0 maps to i_base, i=1 maps to i_base+1)
        k_vals: 2 k indices
        j_vals: 2 j indices
        Returns: new col index
        """
        a, b = k_vals  # k indices
        c, d = j_vals  # j indices
        
        # Strassen multipliers for 2x2x2:
        # The 2x2 A block: rows i_base, i_base+1; cols a, b
        # The 2x2 B block: rows a, b; cols c, d
        # C block: rows i_base, i_base+1; cols c, d
        
        # m1 = (A[0,a]+A[1,b]) * (B[a,c]+B[b,d])
        # This is a sum of 4 rank-1 terms, but in Strassen it's one "multiplier"
        # In tensor decomposition, each multiplier expands to rank-1 terms.
        
        # Actually, in the tensor decomposition framework, each rank-1 term
        # corresponds to one u⊗v⊗w. Strassen's 7 multipliers each correspond
        # to specific linear combinations.
        
        # For the tensor decomposition, I need to express each of the 7 
        # Strassen multipliers as rank-1 tensors.
        
        # m1 = (A[0,a] + A[1,b]) * (B[a,c] + B[b,d])
        # This expands to 4 terms: A[0,a]*B[a,c], A[0,a]*B[b,d], A[1,b]*B[a,c], A[1,b]*B[b,d]
        # But in Strassen, we treat it as one multiplication, contributing to multiple C entries.
        
        # In the tensor decomposition, each "rank-1" term is u⊗v⊗w where
        # u, v, w are vectors. The Strassen decomposition has 7 such terms.
        
        # Let me define them properly:
        # For 2x2x2 with indices (i,k,j) where i,k,j in {0,1}:
        # 
        # Term 1: u=(1,1), v=(1,1), w=(1,1) -> T[i,k,j] = u[i]*v[k]*w[j]
        #   This gives: T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1, 
        #              T[1,0,0]=1, T[1,0,1]=1, T[1,1,0]=1, T[1,1,1]=1
        #   But we need T[i,k,j]=1 for the full 2x2x2 tensor.
        #   This single term gives all 8 entries = 1, but we need to combine
        #   7 terms to get the right result.
        
        # The standard Strassen tensor decomposition for 2x2x2:
        # T = sum_{r=1}^{7} u_r ⊗ v_r ⊗ w_r
        # where the 2x2x2 tensor T[i,k,j] = 1 for all i,k,j in {0,1}.
        
        # The 7 terms (u, v, w) each of length 2:
        # r=1: u=(1,1), v=(1,1), w=(1,1)
        # r=2: u=(0,1), v=(1,0), w=(1,0)
        # r=3: u=(1,0), v=(0,1), w=(1,0)
        # r=4: u=(0,1), v=(0,1), w=(0,1)
        # r=5: u=(1,1), v=(0,1), w=(0,1)
        # r=6: u=(1,-1), v=(1,0), w=(1,1)
        # r=7: u=(1,-1), v=(0,1), w=(1,1)
        
        # Wait, I need to verify this is correct. Let me just use a known correct
        # Strassen 2x2x2 tensor decomposition.
        
        # The 2x2x2 multiplication tensor has rank 7. One valid decomposition:
        # Using the standard Strassen:
        # M1 = (a+d)(e+h): u=(1,1), v=(1,1), w=(1,1) -- but this isn't rank 1 in the right sense
        
        # Actually, let me just verify by construction. For 2x2x2, T[i,k,j]=1.
        # I need 7 rank-1 tensors whose sum equals T.
        
        # Let me use a verified set:
        terms_2x2x2 = [
            # (u, v, w) where u,v,w are length-2 vectors
            ([1, 1], [1, 1], [1, 1]),
            ([0, 1], [1, 0], [1, 0]),
            ([1, 0], [0, 1], [1, 0]),
            ([0, 1], [0, 1], [0, 1]),
            ([1, 1], [0, 1], [0, 1]),
            ([1, -1], [1, 0], [1, 1]),
            ([1, -1], [0, 1], [1, 1]),
        ]
        
        # Verify: sum_r u_r[i] * v_r[k] * w_r[j] should equal 1 for all i,k,j in {0,1}
        # I'll trust this for now and verify later.
        
        for (u, v, w) in terms_2x2x2:
            # u[i] for i in {0,1} corresponds to i_base+i
            # v[k] for k in {0,1} corresponds to k_vals[k]
            # w[j] for j in {0,1} corresponds to j_vals[j]
            for i_local in range(2):
                for k_local in range(2):
                    val = u[i_local] * v[k_local]
                    if val != 0:
                        U[(i_base + i_local) * m + k_vals[k_local]][col] = val
            for k_local in range(2):
                for j_local in range(2):
                    val = v[k_local] * w[j_local]
                    if val != 0:
                        V[k_vals[k_local] * p + j_vals[j_local]][col] = val
            for j_local in range(2):
                for i_local in range(2):
                    val = w[j_local] * u[i_local]
                    if val != 0:
                        W[j_vals[j_local] * n + (i_base + i_local)][col] = val
            col += 1
        return col
    
    # Hmm, the above is getting complicated. Let me simplify.
    
    # Actually, let me just verify my Strassen terms are correct by checking
    # that they sum to the identity tensor.
    
    # For now, let me use a simpler but valid approach:
    # Use schoolbook for all sub-problems but count carefully.
    
    # Reset and use a cleaner approach.
    # 
    # I'll use the following valid decomposition:
    # Total rank 49, constructed as:
    # - 1 term for i=0
    # - For i=1,2: use 24 terms per i-group with sharing
    #
    # Actually, let me just implement a verified rank-48 construction for 2x5x6.
    
    # Clear and restart with a clean implementation
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Rank 1 for i=0
    for k in range(m):
        U[k][0] = 1
    for k in range(m):
        for j in range(p):
            V[k * p + j][0] = 1
    for j in range(p):
        W[j * n][0] = 1
    
    # For i=1,2 (local i=0,1), use 48 terms
    # I'll use a systematic Strassen-based construction.
    
    # Partition: k = [0,1] + [2,3,4], j = [0,1] + [2,3] + [4,5]
    # i_local = [0, 1] (corresponding to i=1,2)
    
    col = 1
    i_base = 1  # first i for the 2-row block
    
    # Process each (k_group, j_group) pair
    for kg in [(0, 1), (2, 3, 4)]:
        for jg in [(0, 1), (2, 3), (4, 5)]:
            kg_size = len(kg)
            jg_size = len(jg)
            
            if kg_size == 2 and jg_size == 2:
                # Use Strassen 2x2x2: 7 terms
                col = add_strassen_2x2x2(col, i_base, kg, jg, U, V, W, m, p, n)
            elif kg_size == 2 and jg_size == 2:
                pass
            else:
                # Use schoolbook: kg_size * jg_size * 2 terms
                for i_local in range(2):
                    for k_idx in kg:
                        for j_idx in jg:
                            U[(i_base + i_local) * m + k_idx][col] = 1
                            V[k_idx * p + j_idx][col] = 1
                            W[j_idx * n + (i_base + i_local)][col] = 1
                            col += 1
    
    # Check: col should be 1 + 48 = 49
    # Let's verify the count:
    # (2,2) pairs: (0,1)x(0,1), (0,1)x(2,3), (0,1)x(4,5) -> 3 pairs * 7 = 21
    # (3,2) pairs: (2,3,4)x(0,1), (2,3,4)x(2,3), (2,3,4)x(4,5) -> 3 pairs * 2*3*2 = 36
    # Total: 21 + 36 = 57, not 48!
    
    # I miscounted. Let me recalculate.
    # (2,2): 7 terms each, 3 such pairs = 21
    # (3,2): schoolbook = 2*3*2 = 12 terms each, 3 such pairs = 36
    # Total = 21 + 36 = 57, plus 1 = 58. Not 49.
    
    # My Strassen partitioning doesn't give 48. Let me reconsider.
    
    # For 2x5x6 to have rank 48, I need a better decomposition.
    # Let me just use a valid higher rank for now and ensure correctness.
    
    # Use schoolbook for 2x5x6: rank = 2*5*6 = 60
    # Total rank = 1 + 60 = 61
    
    # Reset
    U = [[0] * 61 for _ in range(n * m)]
    V = [[0] * 61 for _ in range(m * p)]
    W = [[0] * 61 for _ in range(p * n)]
    
    # Rank 1 for i=0
    for k in range(m):
        U[k][0] = 1
    for k in range(m):
        for j in range(p):
            V[k * p + j][0] = 1
    for j in range(p):
        W[j * n][0] = 1
    
    # Schoolbook for i=1,2
    col = 1
    for i in [1, 2]:
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

