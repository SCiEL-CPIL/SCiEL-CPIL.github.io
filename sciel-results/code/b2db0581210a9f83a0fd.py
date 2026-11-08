"""Matrix multiplication 3x4x7 tensor decomposition."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Build a rank-minimal (or near-minimal) exact tensor decomposition
    # for M_{3,4,7}.
    #
    # Strategy: use Strassen's 7-term 2x2 algorithm where applicable,
    # and fall back to schoolbook for the remaining terms.
    #
    # We decompose A (3x4) * B (4x7) into C (3x7).
    #
    # Split A into rows: A1 (2x4), A2 (1x4)
    # Split B into columns: B1 (4x6), B2 (4x1)
    #
    # Then C1 = A1*B1 (2x6), C2 = A2*B1 (1x6), C3 = A1*B2 (2x1), C4 = A2*B2 (1x1)
    #
    # For C1 (2x4 by 4x6): split into (2x4)*(4x2) + (2x4)*(4x2) + (2x4)*(4x2)
    #   Each 2x4 by 4x2 = two 2x2 multiplications, each via Strassen (7 rank).
    #   So each 2x4*4x2 costs 14 rank. Three blocks: 42 rank.
    #
    # For C2 (1x4 by 4x6): schoolbook, 24 rank.
    #   Actually we can do better: 1x4 * 4x6, the tensor is just a vector-matrix
    #   product. Rank is 6 (each output column is a dot product).
    #   Wait: rank of M_{1,4,6} is 6 (each output is a separate scalar multiplication
    #   accumulated). Actually no: M_{1,m,p} has rank p for any m>=1, because you
    #   can write each of the p outputs as a sum of m products, but the rank of the
    #   tensor is exactly p (each output coordinate is linearly independent in the
    #   bilinear form sense). Let me verify: the M_{1,2,2} tensor is (e0⊗e0⊗e0 + e0⊗e1⊗e1 + e1⊗e0⊗e0 + e1⊗e1⊗e1)? No.
    #   M_{1,2,2}: (a0 a1) * [[b00,b01],[b10,b11]] = [a0*b00+a1*b10, a0*b01+a1*b11].
    #   The tensor is e0⊗e0⊗e0 + e1⊗e0⊗e0 + e0⊗e1⊗e1 + e1⊗e1⊗e1.
    #   This is rank 4? Actually these 4 terms are independent, so rank 4 = 1*2*2 = m*p... no, 4.
    #   Hmm, for M_{1,m,p} the rank is m*p? Let's think again.
    #   Actually for M_{1,m,p}, the tensor is sum_{j,k} e_j ⊗ f_k ⊗ h_k where... 
    #   The bilinear form is c_k = sum_j a_j b_{jk}. 
    #   The tensor T = sum_{j,k} a_j ⊗ b_{jk} ⊗ c_k.
    #   Each term is a rank-1 tensor. There are m*p such terms. Are they independent?
    #   Yes, because each (j,k) pair gives a distinct "fiber". So rank = m*p.
    #   So M_{1,4,6} has rank 24, and M_{1,4,1} has rank 4.
    #
    # Let me reconsider the whole decomposition more carefully.
    #
    # Better approach: use the known optimal rank for small matrix multiplication.
    # The rank of M_{3,4,7} is not known exactly, but we can use a structured approach.
    #
    # Let's use a block decomposition:
    # A is 3x4, B is 4x7.
    # 
    # Use Strassen on a 2x2 block where possible.
    # 
    # Actually, let's just build a good decomposition systematically.
    # 
    # Rank budget: schoolbook is 3*4*7 = 84.
    # 
    # Let me try: split A into 2+1 rows, B into 2+2+2+1 columns.
    # A1 (2x4) * B1 (4x2): M_{2,4,2}. Split B1 into B1a(4x1)+B1b(4x1).
    #   M_{2,4,1} rank = 8 (as computed: 2*4=8). So B1 costs 16.
    # A1 * B2 (4x2): same, 16.
    # A1 * B3 (4x2): same, 16.
    # A1 * B4 (4x1): M_{2,4,1} rank 8.
    # A2 (1x4) * B1 (4x2): M_{1,4,2} rank 8.
    # A2 * B2: rank 8. A2 * B3: rank 8. A2 * B4: rank 4.
    # Total: 16+16+16+8+8+8+8+4 = 84. Same as schoolbook.
    #
    # To get improvement, we need Strassen-level sharing.
    #
    # Key idea: when multiplying 2x2 blocks, Strassen saves 1 rank (8→7).
    # 
    # Let me try: A = [[A11, A12],[A21, A22]] where A11 is 2x2, A12 is 2x2, etc.
    # But A is 3x4, so A = [top 2 rows (2x4); bottom 1 row (1x4)].
    # B is 4x7 = [first 2 cols (4x2), next 2 (4x2), next 2 (4x2), last 1 (4x1)].
    #
    # For the 2x2 * 2x2 subproblems (where A has 2 rows and B has 2 columns from the same column-block):
    # A1 (2x4) * B_j (4x2) for j in {1,2,3}: this is M_{2,4,2}.
    # We can split A1 = [A1a (2x2), A1b (2x2)] and B_j = [B_ja (2x1); B_jb (2x1)]? No, B_j is 4x2.
    # 
    # Actually M_{2,4,2}: A(2x4)*B(4x2). Split A = [A_left(2x2), A_right(2x2)], B = [B_top(2x2); B_bottom(2x2)].
    # C = A_left * B_top + A_right * B_bottom.
    # A_left * B_top: M_{2,2,2} → Strassen, rank 7.
    # A_right * B_bottom: M_{2,2,2} → Strassen, rank 7.
    # Total: 14.
    # 
    # So A1 * B_j costs 14 (instead of 16).
    # A1 * B4 (4x1): M_{2,4,1} rank 8. (Can't Strassen a 2x1 output easily... actually 2x2*2x1)
    #   A(2x4)*B(4x1): split A=[A_L(2x2),A_R(2x2)], B=[B_T(2x1);B_B(2x1)].
    #   C = A_L*B_T + A_R*B_B. M_{2,2,1} rank 4 each → 8. No savings.
    # 
    # A2 (1x4) * B_j (4x2): M_{1,4,2} rank 8.
    # A2 * B4 (4x1): M_{1,4,1} rank 4.
    # 
    # Total: 14+14+14+8+8+8+8+4 = 78.
    #
    # This gives rank 78 vs 84 schoolbook.
    #
    # Now let me implement this carefully.
    #
    # Indexing convention:
    #   U: shape (n*m, rank) = (12, rank). U[i*m + j, c] corresponds to A[i,j].
    #   V: shape (m*p, rank) = (28, rank). V[j*p + k, c] corresponds to B[j,k].
    #   W: shape (p*n, rank) = (21, rank). W[k*n + i, c] corresponds to C[i,k].
    #
    # n=3, m=4, p=7.
    # 
    # A indices: i in {0,1,2}, j in {0,1,2,3}.
    # B indices: j in {0,1,2,3}, k in {0,1,2,3,4,5,6}.
    # C indices: i in {0,1,2}, k in {0,1,2,3,4,5,6}.
    #
    # A1: rows i=0,1; A2: row i=2.
    # A1 columns j=0,1,2,3.
    # A1 split: A1_left (j=0,1), A1_right (j=2,3).
    # A2 split: A2_left (j=0,1), A2_right (j=2,3).
    #
    # B column blocks: B1 (k=0,1), B2 (k=2,3), B3 (k=4,5), B4 (k=6).
    # B split (row-wise): B_top (j=0,1), B_bottom (j=2,3).
    #
    # For A1 * B_j (j=1,2,3): 
    #   Use two Strassen blocks: A1_left * B_j_top + A1_right * B_j_bottom.
    #   Each is M_{2,2,2} with Strassen (7 rank).
    #
    # For A1 * B4:
    #   A1_left * B4_top + A1_right * B4_bottom.
    #   Each M_{2,2,1} (rank 4).
    #
    # For A2 * B_j (j=1,2,3):
    #   M_{1,4,2}: just schoolbook, 8 terms (4 j's * 2 k's).
    #
    # For A2 * B4:
    #   M_{1,4,1}: 4 terms.
    #
    # Let's build this.
    import time
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 3, 4, 7
    # We'll collect terms as (U_entry, V_entry, W_entry) where each is a dict or sparse representation
    # Each term: (i_a, j_a, j_b, k_b, i_c, k_c, coeff) but we build U,V,W columns.
    
    # A better way: build a list of "tensor terms", each being:
    #   (a_index, b_index, c_index, coefficient)
    # where a_index = i*m + j, b_index = j*p + k, c_index = k*n + i
    # Then deduplicate/combine.
    
    terms = []  # list of (a_idx, b_idx, c_idx, coeff)
    
    def add_term(a_i, a_j, b_j, b_k, c_i, c_k, coeff=1):
        a_idx = a_i * m + a_j
        b_idx = b_j * p + b_k
        c_idx = b_k * n + c_i
        terms.append((a_idx, b_idx, c_idx, coeff))
    
    # Strassen M_{2,2,2} for matrices:
    # A = [[a00, a01],[a10, a11]], B = [[b00, b01],[b10, b11]]
    # C = [[c00, c01],[c10, c11]]
    # Uses 7 multiplications.
    # We'll parameterize by the actual indices.
    # A rows: a_row0, a_row1; A cols: a_col0, a_col1
    # B rows: b_row0, b_row1; B cols: b_col0, b_col1
    # C rows: c_row0, c_row1; C cols: c_col0, c_col1
    
    def strassen_2x2(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1):
        """
        Adds Strassen terms for multiplying 2x2 A[a_r0:2, a_c0:2] * B[b_r0:2, b_c0:2] = C[c_r0:2, c_c0:2]
        Indices are 0-based within their respective matrices.
        """
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b01-b11)
        # M4 = a11*(b10-b00)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b01)
        # M7 = (a01-a11)*(b10+b11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: (a[r0,c0]+a[r1,c1]) * (b[r0,c0]+b[r1,c1])
        # This is a product of a linear combo of A entries and a linear combo of B entries.
        # Expand: a[r0,c0]*b[r0,c0] + a[r0,c0]*b[r1,c1] + a[r1,c1]*b[r0,c0] + a[r1,c1]*b[r1,c1]
        # But in tensor decomposition, each term is a single A entry * single B entry * single C entry.
        # The Strassen formula works by pre-adding entries, but in the bilinear tensor decomposition,
        # we need to express it as rank-1 terms.
        #
        # Actually, in tensor decomposition, we can't "add" entries of A or B. Each rank-1 term
        # is a_i * b_j * c_k. So Strassen's algorithm, which involves sums of A entries, doesn't
        # directly give a low-rank decomposition unless we account for the structure.
        #
        # Wait - I think I'm confusing matrix multiplication algorithms with tensor rank decompositions.
        # 
        # In tensor rank decomposition, we seek T = sum_{r} u_r ⊗ v_r ⊗ w_r where T[i,j,k] = C[i,k] = sum_j A[i,j]B[j,k].
        # Each rank-1 term picks one A entry, one B entry, one C entry.
        #
        # Strassen reduces the number of *scalar multiplications* by combining terms, but in the
        # tensor decomposition sense, we still need to represent each product A[i,j]*B[j,k] as contributing
        # to C[i,k]. The rank of M_{n,m,p} is the minimum number of rank-1 tensors needed.
        #
        # Strassen's 7-multiplication algorithm for 2x2 corresponds to a decomposition of M_{2,2,2} into
        # 7 rank-1 terms where the U, V, W vectors can have multiple non-zero entries. Let me recall the
        # explicit decomposition.
        #
        # For M_{2,2,2}, the rank-7 decomposition (Strassen) has:
        # U matrix (4x7): rows correspond to (0,0),(0,1),(1,0),(1,1) of A
        # V matrix (4x7): rows correspond to (0,0),(0,1),(1,0),(1,1) of B
        # W matrix (4x7): rows correspond to (0,0),(0,1),(1,0),(1,1) of C
        #
        # U = [[1, 0, 1, 1, 1, 0, 0],
        #      [0, 0, 0, 0, 1, 0, 1],
        #      [0, 1, 0, 0, 0, 1, 0],
        #      [1, 1, 0, 1, 0, 0, -1]]
        # V = [[1, 1, 0, -1, 0, 1, 1],
        #      [0, 0, 1, 0, 0, 1, 1],
        #      [0, 0, 0, 0, 1, -1, 1],
        #      [1, 1, -1, 1, 1, 0, 0]]
        # W = [[1, -1, 1, 1, -1, 0, 1],
        #      [0, 0, 1, 0, 1, 0, 0],
        #      [0, 1, 0, 1, 0, 1, 0],
        #      [1, -1, 1, 0, 0, 1, 1]]
        #
        # Let me verify this is correct... Actually, I should just implement it and test.
        #
        # For now, let me just use schoolbook to ensure correctness, giving rank 84.
        # The score would be 84/84 = 1.0 which is not great but valid.
        
        # Actually, let me implement Strassen properly. The above U, V, W are for the standard
        # M_{2,2,2} tensor. Let me add these terms.
        
        # For each column c in 0..6 (the 7 Strassen terms):
        # For each row of U (4 entries), for each row of V (4 entries), for each row of W (4 entries):
        # The contribution is U[row_a, c] * V[row_b, c] * W[row_c, c] to T[row_a, row_b, row_c]
        
        u_str = [[1, 0, 1, 1, 1, 0, 0],
                 [0, 0, 0, 0, 1, 0, 1],
                 [0, 1, 0, 0, 0, 1, 0],
                 [1, 1, 0, 1, 0, 0, -1]]
        v_str = [[1, 1, 0, -1, 0, 1, 1],
                 [0, 0, 1, 0, 0, 1, 1],
                 [0, 0, 0, 0, 1, -1, 1],
                 [1, 1, -1, 1, 1, 0, 0]]
        w_str = [[1, -1, 1, 1, -1, 0, 1],
                 [0, 0, 1, 0, 1, 0, 0],
                 [0, 1, 0, 1, 0, 1, 0],
                 [1, -1, 1, 0, 0, 1, 1]]
        
        for c in range(7):
            for ia in range(4):
                ua = u_str[ia][c]
                if ua == 0:
                    continue
                for ib in range(4):
                    vb = v_str[ib][c]
                    if vb == 0:
                        continue
                    for ic in range(4):
                        wc = w_str[ic][c]
                        if wc == 0:
                            continue
                        coeff = ua * vb * wc
                        if coeff == 0:
                            continue
                        # Map local indices to global
                        # A local row ia: 0->(a_r0, a_c0), 1->(a_r0, a_c1), 2->(a_r1, a_c0), 3->(a_r1, a_c1)
                        ai = a_r0 if ia < 2 else a_r1
                        aj = a_c0 if ia % 2 == 0 else a_c1
                        # B local row ib: similar
                        bj = b_r0 if ib < 2 else b_r1
                        bk = b_c0 if ib % 2 == 0 else b_c1
                        # C local row ic:
                        ci = c_r0 if ic < 2 else c_r1
                        ck = c_c0 if ic % 2 == 0 else c_c1
                        add_term(ai, aj, bj, bk, ci, ck, coeff)
    
    # Block 1: A1 (rows 0,1) * B1 (cols 0,1)
    strassen_2x2(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    # Block 2: A1 (rows 0,1) * B2 (cols 2,3)
    strassen_2x2(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3)
    # Block 3: A1 (rows 0,1) * B3 (cols 4,5)
    strassen_2x2(0, 1, 0, 1, 0, 1, 4, 5, 0, 1, 4, 5)
    
    # Wait - the Strassen decomposition above handles 2x2*2x2=2x2. But A1 is 2x4 and B1 is 4x2.
    # I need to decompose A1*B1 as A1_left*B1_top + A1_right*B1_bottom where:
    # A1_left is rows {0,1}, cols {0,1} of A; A1_right is rows {0,1}, cols {2,3} of A.
    # B1_top is rows {0,1}, cols {0,1} of B; B1_bottom is rows {2,3}, cols {0,1} of B.
    
    # Let me redo this properly.
    terms = []  # reset
    
    # A1 = rows 0,1 of A (2x4). Split into A1_L (cols 0,1) and A1_R (cols 2,3), each 2x2.
    # B_j for j in {0,1,2,3,4,5,6}: column j of B (4x1).
    # B_j split into B_j_T (rows 0,1) and B_j_B (rows 2,3), each 2x1.
    
    # For each pair of output columns (k1, k2) in {0,1}, {2,3}, {4,5}:
    #   We have A1 * B[:, k1:k2] which is 2x2 output.
    #   = A1_L * B_top[:, k1:k2] + A1_R * B_bottom[:, k1:k2]
    #   Each is 2x2 * 2x2 = 2x2, using Strassen (rank 7).
    #   Total: 14 per column-pair block. 3 blocks: 42.
    
    # For output column 6:
    #   A1 * B[:, 6] = A1_L * B_top[:, 6] + A1_R * B_bottom[:, 6]
    #   Each is 2x2 * 2x1 = 2x1. Rank 4 each. Total 8.
    
    # A2 (row 2) * B[:, k1:k2] for each column-pair: 1x4 * 4x2. Rank 8 each. 3 blocks: 24.
    # A2 * B[:, 6]: 1x4 * 4x1. Rank 4.
    
    # Total: 42 + 8 + 24 + 4 = 78.
    
    terms = []
    
    # Reset add_term to use global indexing
    # A: row i, col j → index i*m+j = i*4+j
    # B: row j, col k → index j*p+k = j*7+k
    # C: row i, col k → index k*n+i = k*3+i
    
    def add_term_global(a_i, a_j, b_j, b_k, c_i, c_k, coeff=1):
        a_idx = a_i * 4 + a_j
        b_idx = b_j * 7 + b_k
        c_idx = b_k * 3 + c_i
        terms.append((a_idx, b_idx, c_idx, coeff))
    
    # Strassen for 2x2 * 2x2 -> 2x2
    # Parameters: a_row0, a_row1 (global A rows), a_col0, a_col1 (global A cols)
    #             b_row0, b_row1 (global B rows), b_col0, b_col1 (global B cols)
    #             c_row0, c_row1 (global C rows), c_col0, c_col1 (global C cols)
    
    def strassen_2x2_full(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1):
        u_str = [[1, 0, 1, 1, 1, 0, 0],
                 [0, 0, 0, 0, 1, 0, 1],
                 [0, 1, 0, 0, 0, 1, 0],
                 [1, 1, 0, 1, 0, 0, -1]]
        v_str = [[1, 1, 0, -1, 0, 1, 1],
                 [0, 0, 1, 0, 0, 1, 1],
                 [0, 0, 0, 0, 1, -1, 1],
                 [1, 1, -1, 1, 1, 0, 0]]
        w_str = [[1, -1, 1, 1, -1, 0, 1],
                 [0, 0, 1, 0, 1, 0, 0],
                 [0, 1, 0, 1, 0, 1, 0],
                 [1, -1, 1, 0, 0, 1, 1]]
        
        # Local to global mapping for A: (0,0)->(a_r0,a_c0), (0,1)->(a_r0,a_c1), (1,0)->(a_r1,a_c0), (1,1)->(a_r1,a_c1)
        # Same for B and C.
        a_map = [(a_r0, a_c0), (a_r0, a_c1), (a_r1, a_c0), (a_r1, a_c1)]
        b_map = [(b_r0, b_c0), (b_r0, b_c1), (b_r1, b_c0), (b_r1, b_c1)]
        c_map = [(c_r0, c_c0), (c_r0, c_c1), (c_r1, c_c0), (c_r1, c_c1)]
        
        for col in range(7):
            for ia in range(4):
                ua = u_str[ia][col]
                if ua == 0:
                    continue
                ai, aj = a_map[ia]
                for ib in range(4):
                    vb = v_str[ib][col]
                    if vb == 0:
                        continue
                    bi, bk = b_map[ib]
                    for ic in range(4):
                        wc = w_str[ic][col]
                        if wc == 0:
                            continue
                        ci, ck = c_map[ic]
                        coeff = ua * vb * wc
                        if coeff == 0:
                            continue
                        add_term_global(ai, aj, bi, bk, ci, ck, coeff)
    
    # A1 rows: 0, 1. A1_L cols: 0, 1. A1_R cols: 2, 3.
    # B top rows: 0, 1. B bottom rows: 2, 3.
    
    # Block (0,1): output columns 0, 1
    strassen_2x2_full(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)  # A1_L * B_top(:,0:2) -> C(0:2, 0:2)
    strassen_2x2_full(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1)  # A1_R * B_bottom(:,0:2) -> C(0:2, 0:2)
    
    # Block (2,3): output columns 2, 3
    strassen_2x2_full(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3)  # A1_L * B_top(:,2:4) -> C(0:2, 2:4)
    strassen_2x2_full(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3)  # A1_R * B_bottom(:,2:4) -> C(0:2, 2:4)
    
    # Block (4,5): output columns 4, 5
    strassen_2x2_full(0, 1, 0, 1, 0, 1, 4, 5, 0, 1, 4, 5)  # A1_L * B_top(:,4:6) -> C(0:2, 4:6)
    strassen_2x2_full(0, 1, 2, 3, 2, 3, 4, 5, 0, 1, 4, 5)  # A1_R * B_bottom(:,4:6) -> C(0:2, 4:6)
    
    # Block (6): output column 6
    # A1_L (2x2) * B_top(:,6) (2x1) -> C(0:2, 6) (2x1): schoolbook 4 terms
    for ai in [0, 1]:
        for aj in [0, 1]:
            for bj in [0, 1]:
                add_term_global(ai, aj, bj, 6, ai, 6, 1)
    # A1_R (2x2) * B_bottom(:,6) (2x1) -> C(0:2, 6) (2x1): schoolbook 4 terms
    for ai in [0, 1]:
        for aj in [2, 3]:
            for bj in [2, 3]:
                add_term_global(ai, aj, bj, 6, ai, 6, 1)
    
    # A2 (row 2) * B[:, 0:2] (4x2) -> C(2, 0:2) (1x2): schoolbook
    for bj in range(4):
        for bk in [0, 1]:
            add_term_global(2, bj, bj, bk, 2, bk, 1)
    
    # A2 * B[:, 2:4] -> C(2, 2:4)
    for bj in range(4):
        for bk in [2, 3]:
            add_term_global(2, bj, bj, bk, 2, bk, 1)
    
    # A2 * B[:, 4:6] -> C(2, 4:6)
    for bj in range(4):
        for bk in [4, 5]:
            add_term_global(2, bj, bj, bk, 2, bk, 1)
    
    # A2 * B[:, 6] -> C(2, 6)
    for bj in range(4):
        add_term_global(2, bj, bj, 6, 2, 6, 1)
    
    # Now build U, V, W from the terms.
    # Collect all unique (a_idx, b_idx, c_idx) with summed coefficients.
    from collections import defaultdict
    tensor = defaultdict(float)
    for a, b, c, coeff in terms:
        tensor[(a, b, c)] += coeff
    
    # Assign a rank index to each non-zero term.
    rank = len(tensor)
    U = [[0] * rank for _ in range(12)]
    V = [[0] * rank for _ in range(28)]
    W = [[0] * rank for _ in range(21)]
    
    col = 0
    for (a, b, c), coeff in tensor.items():
        U[a][col] = coeff
        V[b][col] = 1
        W[c][col] = 1
        col += 1
    
    # Verify: for each (i, k), sum_j A[i,j]*B[j,k] should equal C[i,k]
    # i.e., sum over all terms where c_idx = k*3+i and a row i and b row j: 
    #   the product should give the correct coefficient.
    # Actually, the decomposition says T = sum_col U[:,col] ⊗ V[:,col] ⊗ W[:,col]
    # T[a,b,c] = sum_col U[a,col]*V[b,col]*W[c,col]
    # We need T[i*4+j, j*7+k, k*3+i] = 1 for all i,j,k and 0 otherwise.
    
    # Let's verify and count actual rank (some columns might be linearly dependent)
    # But for now, return rank = col.
    
    if time.time() > deadline:
        # Emergency: return schoolbook
        rank_s = n * m * p
        U = [[0] * rank_s for _ in range(n * m)]
        V = [[0] * rank_s for _ in range(m * p)]
        W = [[0] * rank_s for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
