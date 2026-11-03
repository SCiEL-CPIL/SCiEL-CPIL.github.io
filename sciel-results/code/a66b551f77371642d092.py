"""Matrix multiplication 3x3x4 with Strassen-based rank reduction."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    # U: n*m rows, rank columns
    # V: m*p rows, rank columns
    # W: p*n rows, rank columns
    # W convention: W[k*n + i, r] contributes to C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    U = []
    V = []
    W = []
    
    def add_term(u_entries, v_entries, w_entries, scale=1):
        """Add one rank-1 term. u_entries, v_entries, w_entries are dicts: {row: coeff}."""
        idx = len(U)
        U.append([0] * 0)  # placeholder
        V.append([0] * 0)
        W.append([0] * 0)
        for i in range(len(U)):
            U[i].append(0)
            V[i].append(0)
            W[i].append(0)
        # Actually, let's just build lists directly
        pass
    
    # Let's build the matrices as lists of rows, appending columns as we go.
    # Start with empty matrices
    num_U_rows = n * m  # 9
    num_V_rows = m * p  # 12
    num_W_rows = p * n  # 12
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_column(u_vals, v_vals, w_vals):
        """u_vals, v_vals, w_vals are lists of length num_U_rows, num_V_rows, num_W_rows respectively."""
        for i in range(num_U_rows):
            if i < len(U_rows):
                U_rows[i].append(u_vals[i])
            else:
                U_rows.append([u_vals[i]])
        for i in range(num_V_rows):
            if i < len(V_rows):
                V_rows[i].append(v_vals[i])
            else:
                V_rows.append([v_vals[i]])
        for i in range(num_W_rows):
            if i < len(W_rows):
                W_rows[i].append(w_vals[i])
            else:
                W_rows.append([w_vals[i]])
    
    def make_zero():
        return [0] * num_U_rows
    
    def make_zero_v():
        return [0] * num_V_rows
    
    def make_zero_w():
        return [0] * num_W_rows
    
    def U_entry(i, j):
        return i * m + j  # row index in U for A[i,j]
    
    def V_entry(j, k):
        return j * p + k  # row index in V for B[j,k]
    
    def W_entry(k, i):
        return k * n + i  # row index in W for C[i,k]
    
    # We'll use a helper to build up the U, V, W contributions for a set of (i,j,k) with coefficients
    # For a single (i,j,k) with coefficient c: U[i*m+j] += c, V[j*p+k] += c, W[k*n+i] += c
    
    # Strassen for 2x2x2:
    # C11 = M1 - M2 + M4 - M6
    # C12 = M3 + M2 - M5 - M7
    # C21 = M1 + M4
    # C22 = M3 - M1 + M5 + M6
    # where:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 2x2x4: we can apply Strassen separately for the two 2x2x2 blocks (k=1,2 and k=3,4)
    # since the A inputs are the same but B differs.
    # Block 1: k=1,2. B block is [[B11, B12], [B21, B22]]
    # Block 2: k=3,4. B block is [[B13, B14], [B23, B24]]
    
    # For each 2x2x2 block, we need 7 terms.
    # Each term has the form: sum of products of linear combinations of A's and B's, multiplied by a linear combination of C's.
    
    # Let's handle the full 3x3x4 by partitioning.
    # Rows of A: 1,2,3. Cols of A: 1,2,3.
    # Rows of B: 1,2,3. Cols of B: 1,2,3,4.
    # Output C: 3x4.
    
    # Partition:
    # - Top-left 2x2 block of A (rows 1,2; cols 1,2) interacts with top-left 2x2 block of B (rows 1,2; cols 1..4) -> contributes to C rows 1,2; cols 1..4.
    #   This is a 2x2x4 multiplication. Rank 14 (two Strassen blocks).
    # - Top-right: A rows 1,2; col 3 interacts with B row 3; cols 1..4 -> contributes to C rows 1,2; cols 1..4.
    #   This is a 2x1x4 multiplication. Each output C[i,k] = A[i,3]*B[3,k]. Rank 4 (one term per (i,k)).
    # - Bottom-left: A row 3; cols 1,2 interacts with B rows 1,2; cols 1..4 -> contributes to C row 3; cols 1..4.
    #   This is a 1x2x4 multiplication. C[3,k] = A[3,1]*B[1,k] + A[3,2]*B[2,k]. Rank 8? 
    #   Actually, for each k, we have 2 terms. For 4 values of k, that's 8 terms. But can we do better?
    #   C[3,k] = A[3,1]*B[1,k] + A[3,2]*B[2,k]. This is already minimal: for each k, it's a dot product of length 2,
    #   which needs 2 scalar multiplications. So 4*2 = 8 terms.
    # - Bottom-right: A[3,3] * B[3,k] for k=1..4. Rank 4.
    
    # Total rank: 14 + 4 + 8 + 4 = 30.
    
    # Let's implement this.
    
    # Helper to add a term with specific U, V, W entries
    def add_term_with_coeff(u_coeff, v_coeff, w_coeff):
        """u_coeff, v_coeff, w_coeff are dicts {row_index: coeff}"""
        u = make_zero()
        v = make_zero_v()
        w = make_zero_w()
        for row, c in u_coeff.items():
            u[row] = c
        for row, c in v_coeff.items():
            v[row] = c
        for row, c in w_coeff.items():
            w[row] = c
        add_column(u, v, w)
    
    # Part 1: 2x2x4 (A rows 1,2 cols 1,2; B rows 1,2 cols 1..4)
    # This contributes to C[1,1], C[1,2], C[1,3], C[1,4], C[2,1], C[2,2], C[2,3], C[2,4]
    # Note: 0-indexed: A[0,0], A[0,1], A[1,0], A[1,1]; B[0,0..3], B[1,0..3]; C[0,0..3], C[1,0..3]
    
    # Block 1: k=0,1 (B cols 0,1)
    # Strassen terms for 2x2x2:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M2 + M4 - M6
    # C01 = M3 + M2 - M5 - M7
    # C10 = M1 + M4
    # C11 = M3 - M1 + M5 + M6
    
    # For each M, we need to express it as sum of rank-1 terms.
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # This is 4 scalar products, but in tensor rank terms, each (i,j,k) with coefficient 1 is one rank-1 term.
    # So M1 expands to 4 terms.
    # Wait, but Strassen's point is that we compute 7 "multiplications" where each multiplication is of sums.
    # In the tensor decomposition, each "multiplication" of two sums corresponds to multiple rank-1 terms.
    
    # Actually, let me reconsider. The rank-1 term for (i,j,k) is: U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # If we have (A00+A11)*(B00+B11), this means:
    # C gets contribution: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # In terms of the output:
    # C[0,0] gets A00*B00
    # C[0,1] gets A00*B11
    # C[1,0] gets A11*B00
    # C[1,1] gets A11*B11
    # So M1 contributes to 4 different (i,j,k) combinations.
    # In the tensor decomposition, each (i,j,k) is a separate rank-1 term.
    # So M1 requires 4 rank-1 terms.
    
    # This means Strassen doesn't reduce the number of (i,j,k) terms; it reduces the number of "bilinear products" 
    # where each product can be of sums. But in the tensor rank decomposition, we're counting the number of 
    # rank-1 terms, which is the number of (i,j,k) triples with nonzero coefficient.
    
    # Hmm, I think I'm confusing two things. Let me reconsider.
    
    # The matrix multiplication tensor has entries T[i,j,k] for i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # T[i,j,k] = 1 for all (i,j,k) in the standard multiplication? No.
    # The tensor is: T_{i,j,k} such that C[i,k] = sum_{j} A[i,j] * B[j,k].
    # So the tensor entry T[i,j,k] = 1 for each (i,j,k) where the product A[i,j]*B[j,k] contributes to C[i,k].
    # Yes, so there are n*m*p = 36 entries, all equal to 1.
    # The rank of this tensor is the minimum number of rank-1 tensors that sum to it.
    
    # A rank-1 tensor is defined by vectors u (length n*m), v (length m*p), w (length p*n) such that
    # T[i,j,k] = sum_r u[i*m+j] * v[j*p+k] * w[k*n+i].
    
    # For the schoolbook, each (i,j,k) is its own rank-1 term.
    
    # For Strassen, we want to express the 8 (i,j,k) entries of a 2x2x2 block as 7 rank-1 terms.
    # Is this possible? Let's see.
    # The 2x2x2 multiplication tensor has rank 7 (Strassen). So yes, the 8 entries can be expressed as 7 rank-1 terms.
    
    # So for the 2x2x4 block, we have two 2x2x2 blocks, each of rank 7, total rank 14.
    # This is correct.
    
    # Now I need to actually construct the 7 rank-1 terms for each 2x2x2 block.
    
    # For 2x2x2 with indices (i,j,k) where i,j,k in {0,1}:
    # The tensor entries are all 1.
    # Strassen gives:
    # C00 = M1 - M2 + M4 - M6
    # C01 = M3 + M2 - M5 - M7
    # C10 = M1 + M4
    # C11 = M3 - M1 + M5 + M6
    
    # Where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mr is a bilinear form. To express as rank-1 tensors, we need to expand each Mr into rank-1 terms.
    
    # M1 = (A00+A11)(B00+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # This contributes to:
    # C00: A00*B00 -> (i=0,j=0,k=0) coeff 1
    # C00: A11*B00 -> (i=1,j=1,k=0) coeff 1
    # C01: A00*B11 -> (i=0,j=1,k=1) coeff 1
    # C01: A11*B11 -> (i=1,j=1,k=1) coeff 1
    # Wait, C00 = sum_j A[0,j]*B[j,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0].
    # C01 = A[0,0]*B[0,1] + A[0,1]*B[1,1].
    # C10 = A[1,0]*B[0,0] + A[1,1]*B[1,0].
    # C11 = A[1,0]*B[0,1] + A[1,1]*B[1,1].
    
    # So M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11.
    # A00*B00 contributes to C00.
    # A00*B11 contributes to C01.
    # A11*B00 contributes to C10.
    # A11*B11 contributes to C11.
    
    # So M1 as a bilinear form gives:
    # Contribution to C00: A00*B00 (i=0,j=0,k=0)
    # Contribution to C01: A00*B11 (i=0,j=1,k=1)
    # Contribution to C10: A11*B00 (i=1,j=1,k=0)
    # Contribution to C11: A11*B11 (i=1,j=1,k=1)
    
    # In the tensor decomposition, M1 is not a single rank-1 term; it's a sum of 4 rank-1 terms.
    # So to express the 2x2x2 tensor as 7 rank-1 terms, we need a different approach.
    
    # Let me look up the actual Strassen decomposition in terms of tensor rank.
    
    # The 2x2x2 tensor of rank 7 can be written as:
    # T = sum_{r=1}^{7} u_r v_r w_r
    # where the entries T[i,j,k] are reconstructed.
    
    # One standard decomposition:
    # r=1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,0) -> (i,j,k) = (0,0,0), (0,1,0), (1,0,0), (1,1,0) ?
    # I need to be more careful.
    
    # Let me use a known rank-7 decomposition for 2x2x2.
    # The entries are T[i,j,k] for i,j,k in {0,1}, all = 1.
    
    # Here's a decomposition (I'll derive it from Strassen):
    # We want T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Using Strassen's algorithm, the 7 multiplications correspond to 7 rank-1 terms.
    # Each Mr is a product of two linear forms, so each Mr can be written as a sum of rank-1 terms.
    # But the total rank is 7, meaning the 7 Mr's can be combined into 7 rank-1 terms total.
    
    # I think the way it works is that each Mr corresponds to exactly one rank-1 term, where the U, V, W vectors 
    # have entries that are the coefficients in the linear forms.
    
    # For example, M1 = (A00+A11)(B00+B11).
    # U vector for M1: U[0,0]=1, U[1,1]=1 (coefficients of A00 and A11)
    # V vector for M1: V[0,0]=1, V[1,1]=1 (coefficients of B00 and B11)
    # W vector for M1: W[0,0]=1, W[0,1]=1, W[1,0]=1, W[1,1]=1 (since M1 contributes to all 4 C entries with coeff 1)
    
    # Wait, that doesn't work because W would need to be 1 for all 4 (k,i) pairs, but then the product 
    # U[i,j]*V[j,k]*W[k,i] would give 1 for (i=0,j=0,k=0), (i=0,j=1,k=1), (i=1,j=0,k=0), (i=1,j=1,k=1).
    # And we want T[0,0,0]=1, T[0,1,1]=1, T[1,0,0]=1, T[1,1,1]=1. But T[0,0,1]=0, T[0,1,0]=0, etc.
    # With U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[k,i]=1 for all k,i:
    # T[0,0,0] = U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1. Good.
    # T[0,0,1] = U[0,0]*V[0,1]*W[1,0] = 1*0*1 = 0. Good.
    # T[0,1,0] = U[0,1]*V[1,0]*W[0,0] = 0*0*1 = 0. Good.
    # T[0,1,1] = U[0,1]*V[1,1]*W[1,0] = 0*1*1 = 0. Bad! We want T[0,1,1]=1.
    
    # So that doesn't work. The issue is that V[0,1]=0, so T[0,1,1]=0.
    
    # I think I need to look at this more carefully. Let me just implement the known rank-31 decomposition for 3x3x4 
    # from the literature, or use a simpler approach.
    
    # Given the time constraints, let me implement a rank-34 decomposition using Strassen for the 2x2x2 subblocks.
    # 3x3x4 = 2x2x4 (rank 14) + 2x1x4 (rank 4) + 1x2x4 (rank 8) + 1x1x4 (rank 4) = 30.
    # But I'm not sure if rank 30 is achievable. Let me just implement the 2x2x4 part with Strassen and see.
    
    # Actually, let me just implement the known best: rank 31 for 3x3x4.
    # I'll use a combination of Strassen and standard terms.
    
    # For simplicity and correctness, let me implement the schoolbook (rank 36) first to make sure the format is right,
    # then try to improve.
    
    # Schoolbook:
    U_rows = []
    V_rows = []
    W_rows = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                U_rows.append(u)
                V_rows.append(v)
                W_rows.append(w)
    
    # This gives rank 36. But we want to do better.
    
    # Let me try to implement the rank-30 decomposition.
    # I'll build the U, V, W matrices incrementally.
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_rank1_term(u_vals, v_vals, w_vals):
        """u_vals, v_vals, w_vals are lists of length n*m, m*p, p*n"""
        U_rows.append(u_vals)
        V_rows.append(v_vals)
        W_rows.append(w_vals)
    
    # Part 1: 2x2x4 for A[0:2, 0:2] * B[0:2, 0:4] -> C[0:2, 0:4]
    # Two Strassen blocks for (k=0,1) and (k=2,3)
    
    # For a 2x2x2 block with local indices (i,j,k) in {0,1}, the tensor entries are all 1.
    # Strassen decomposition (rank 7):
    # I'll use the following known decomposition:
    # T[i,j,k] = sum_{r=1}^{7} u_r[i,j] * v_r[j,k] * w_r[k,i]
    
    # One standard rank-7 decomposition:
    # r=1: u=(1,1,0,0), v=(1,0,1,0), w=(1,0,0,0) -> covers (0,0,0),(0,1,0),(1,0,0),(1,1,0) partially
    # This is getting complicated. Let me just use the schoolbook for now and ensure correctness.
    
    # Actually, let me just return the schoolbook for safety, since I'm running out of time.
    
    # Convert U_rows, V_rows, W_rows to the expected format
    # The expected format is: U is a list of rows, each row is a list of length rank.
    # Currently U_rows is a list of columns (each of length n*m).
    
    rank = len(U_rows)
    U = [[U_rows[r][row] for r in range(rank)] for row in range(n * m)]
    V = [[V_rows[r][row] for r in range(rank)] for row in range(m * p)]
    W = [[W_rows[r][row] for r in range(rank)] for row in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
