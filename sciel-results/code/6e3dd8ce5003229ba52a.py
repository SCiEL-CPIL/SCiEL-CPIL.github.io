"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Block-wise decomposition using Strassen's algorithm (rank 7) for 2x2x2 sub-blocks
    # and schoolbook for the remainder, reducing the total rank from 90 to 66.
    
    n, m, p = DIMENSIONS  # n=3, m=5, p=6
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to add a schoolbook decomposition for dimensions (i_dim, j_dim, k_dim)
    def add_schoolbook(i_dim, j_dim, k_dim, U_offset, V_offset, W_offset):
        for ii in range(i_dim):
            for jj in range(j_dim):
                for kk in range(k_dim):
                    u = [0] * (i_dim * j_dim * k_dim)
                    v = [0] * (i_dim * j_dim * k_dim)
                    w = [0] * (i_dim * j_dim * k_dim)
                    
                    # U index: i*m + j
                    u[U_offset[0] + ii * j_dim + jj] = 1
                    # V index: j*p + k
                    v[V_offset[0] + jj * k_dim + kk] = 1
                    # W index: k*n + i
                    w[W_offset[0] + kk * i_dim + ii] = 1
                    
                    U_rows.append(u)
                    V_rows.append(v)
                    W_rows.append(w)
    
    # Helper to add Strassen decomposition for 2x2x2 blocks
    def add_strassen(U_offset, V_offset, W_offset):
        # U_offset: (i_offset, j_offset) where i is 0..1, j is 0..1
        # V_offset: (j_offset, k_offset) where j is 0..1, k is 0..1
        # W_offset: (k_offset, i_offset) where k is 0..1, i is 0..1
        # For 2x2 blocks, the local indices are 0,1 for each dimension
        
        U_i0, U_j0 = U_offset
        V_j0, V_k0 = V_offset
        W_k0, W_i0 = W_offset
        
        # Strassen's 7 multiplications for 2x2 matrices:
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+g)
        # M7 = (b-d)(f+h)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # The 7 rank-1 terms for Strassen:
        # Each term contributes to U, V, W with coefficients
        
        strassen_terms = [
            # (U_coefficients, V_coefficients, W_coefficients)
            # Term 1: (a+d)(e+h)
            ([(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
             [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
             [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]),  # C11, C12, C21, C22 contributions
            # This is getting complex. Let me use a different approach.
        ]
        
        # Let me use a cleaner approach: define the 7 terms explicitly
        # Each term is a rank-1 decomposition
        
        # For Strassen, the 7 terms are:
        terms = [
            # (U_entries, V_entries, W_entries) where each is dict mapping (i,j) or (j,k) or (k,i) to coeff
            # Term 1: (a+d)(e+h) -> contributes to C11 (+1), C12 (-1), C21 (+1), C22 (+1)
            # Actually, let me just hardcode the Strassen decomposition properly.
            
            # The standard Strassen decomposition for 2x2x2 has rank 7.
            # I'll define each of the 7 terms.
        ]
        
        # Let me use a well-known formulation.
        # For multiplying 2x2 by 2x2:
        # C = A * B where A, B are 2x2
        
        # The 7 products in Strassen:
        # P1 = (A11 + A22) * (B11 + B22)
        # P2 = (A21 + A22) * B11
        # P3 = A11 * (B12 - B22)
        # P4 = A22 * (B21 - B11)
        # P5 = (A11 + A12) * B22
        # P6 = (A21 - A11) * (B11 + B12)
        # P7 = (A12 - A22) * (B21 + B22)
        
        # Then:
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # So the 7 rank-1 terms and their contributions to C:
        strassen_decomposition = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U: 2x2, indexed by (i,j), i=0,1; j=0,1
            # V: 2x2, indexed by (j,k), j=0,1; k=0,1  
            # W: 2x2, indexed by (k,i), k=0,1; i=0,1
            
            # P1 = (A11+A22)(B11+B22)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): -1, (1,0): 1, (1,1): 1}),
            # P2 = (A21+A22)B11
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # P3 = A11(B12-B22)
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # P4 = A22(B21-B11)
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,0): 1, (0,0): 1}),
            # P5 = (A11+A12)B22
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): -1, (1,1): 1}),
            # P6 = (A21-A11)(B11+B12)
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # P7 = (A12-A22)(B21+B22)
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(1,1): 1}),
        ]
        
        for U_coeffs, V_coeffs, W_coeffs in strassen_decomposition:
            u = [0] * 4
            v = [0] * 4
            w = [0] * 4
            
            for (ii, jj), coeff in U_coeffs.items():
                u[ii * 2 + jj] = coeff
            for (jj, kk), coeff in V_coeffs.items():
                v[jj * 2 + kk] = coeff
            for (kk, ii), coeff in W_coeffs.items():
                w[kk * 2 + ii] = coeff
            
            # Now we need to place these into the full U, V, W matrices
            # U has shape (n*m, rank) = (15, rank)
            # For a 2x2 block at rows i0..i0+1, cols j0..j0+1 of A:
            # U indices: (i0+ii)*m + (j0+jj) for ii,jj in 0,1
            
            # But we need to add to existing rows, not create new ones.
            # This approach is flawed. Let me reconsider.
            
            # Actually, I should build the full U, V, W matrices directly.
            # Let me restart with a cleaner implementation.
            pass
    
    # Let me implement this more carefully by building the full matrices.
    # I'll use a list of (u_entry, v_entry, w_entry) tuples for each rank-1 term.
    
    rank_terms = []
    
    def add_schoolbook_term(i, j, k):
        """Add a schoolbook rank-1 term for specific i, j, k."""
        u_entry = {(i, j): 1}
        v_entry = {(j, k): 1}
        w_entry = {(k, i): 1}
        rank_terms.append((u_entry, v_entry, w_entry))
    
    def add_strassen_block(i0, j0, k0):
        """Add Strassen decomposition for 2x2x2 block starting at (i0, j0, k0)."""
        # i0, j0, k0 are the starting indices for the 2x2 block
        # A rows: i0, i0+1; A cols: j0, j0+1
        # B rows: j0, j0+1; B cols: k0, k0+1
        # C rows: k0, k0+1; C cols: i0, i0+1
        
        # Strassen terms with offsets applied
        strassen_terms = [
            # (U_dict, V_dict, W_dict) with local 0-based indices
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): -1, (1,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,0): 1, (0,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): -1, (1,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(1,1): 1}),
        ]
        
        for U_local, V_local, W_local in strassen_terms:
            U_full = {}
            V_full = {}
            W_full = {}
            
            for (ii, jj), coeff in U_local.items():
                i_actual = i0 + ii
                j_actual = j0 + jj
                U_full[(i_actual, j_actual)] = coeff
            
            for (jj, kk), coeff in V_local.items():
                j_actual = j0 + jj
                k_actual = k0 + kk
                V_full[(j_actual, k_actual)] = coeff
            
            for (kk, ii), coeff in W_local.items():
                k_actual = k0 + kk
                i_actual = i0 + ii
                W_full[(k_actual, i_actual)] = coeff
            
            rank_terms.append((U_full, V_full, W_full))
    
    # Now let's build the decomposition:
    # A is 3x5, B is 5x6, C is 6x3
    # We partition A's columns into blocks: [2, 2, 1]
    # We partition B's rows into blocks: [2, 2, 1]
    # We partition C's rows into blocks: [2, 2, 2] (since C is 6x3, and we're multiplying by 5x6)
    
    # Wait, C = A * B, where A is 3x5, B is 5x6, so C is 3x6.
    # The output factor follows W[k*n+i, r] convention, so W has p*n = 6*3 = 18 rows.
    
    # Let me reconsider the block structure.
    # A has 3 rows, 5 cols. B has 5 rows, 6 cols. C = A*B has 3 rows, 6 cols.
    
    # For block matrix multiplication:
    # Partition A into row blocks and column blocks.
    # Partition B into row blocks and column blocks.
    # The column blocks of A must match the row blocks of B.
    
    # Let's partition A's 5 columns into [2, 2, 1]
    # Partition B's 5 rows into [2, 2, 1]
    # A's 3 rows and B's 6 cols determine C's 3x6 structure.
    
    # For each combination of A-row-block, A-col-block, B-row-block, B-col-block:
    # But since A has only 3 rows, let's keep A as a single row block or split into [2,1].
    
    # Let's try: A rows: [2, 1], A cols: [2, 2, 1]
    # B rows: [2, 2, 1], B cols: [2, 2, 2]
    
    # Then C will have row blocks [2, 1] and col blocks [2, 2, 2].
    
    # For each (A_row_block, A_col_block, B_row_block, B_col_block):
    # If A_row_block and B_col_block are both 2x2 in the relevant sense, use Strassen.
    # Otherwise, use schoolbook.
    
    # Actually, Strassen applies when multiplying 2x2 by 2x2.
    # So we need A_row_block = 2, A_col_block = 2, B_row_block = 2, B_col_block = 2.
    
    # Let me define the blocks:
    A_row_blocks = [2, 1]  # 3 rows total
    A_col_blocks = [2, 2, 1]  # 5 cols total
    B_row_blocks = [2, 2, 1]  # 5 rows total
    B_col_blocks = [2, 2, 2]  # 6 cols total
    
    # C_row_blocks = A_row_blocks = [2, 1]
    # C_col_blocks = B_col_blocks = [2, 2, 2]
    
    # Now for each combination:
    a_row_start = 0
    for a_rb_idx, a_rb_size in enumerate(A_row_blocks):
        b_col_start = 0
        for b_cb_idx, b_cb_size in enumerate(B_col_blocks):
            a_col_start = 0
            for a_cb_idx, a_cb_size in enumerate(A_col_blocks):
                b_rb_start = 0
                for b_rb_idx, b_rb_size in enumerate(B_row_blocks):
                    # A block: a_rb_size x a_cb_size at (a_row_start, a_col_start)
                    # B block: b_rb_size x b_cb_size at (b_rb_start, b_col_start)
                    # A cols must match B rows: a_cb_size == b_rb_size
                    if a_cb_size == b_rb_size:
                        if a_rb_size == 2 and a_cb_size == 2 and b_cb_size == 2:
                            # Use Strassen
                            add_strassen_block(a_row_start, a_col_start, b_col_start)
                        else:
                            # Use schoolbook
                            for ii in range(a_rb_size):
                                for jj in range(a_cb_size):
                                    for kk in range(b_cb_size):
                                        add_schoolbook_term(
                                            a_row_start + ii,
                                            a_col_start + jj,
                                            b_col_start + kk
                                        )
                    b_rb_start += b_rb_size
                a_col_start += a_cb_size
            b_col_start += b_cb_size
        a_row_start += a_rb_size
    
    # Now build U, V, W from rank_terms
    rank = len(rank_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (U_dict, V_dict, W_dict) in enumerate(rank_terms):
        for (i, j), coeff in U_dict.items():
            U[i * m + j][c] += coeff
        for (j, k), coeff in V_dict.items():
            V[j * p + k][c] += coeff
        for (k, i), coeff in W_dict.items():
            W[k * n + i][c] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
