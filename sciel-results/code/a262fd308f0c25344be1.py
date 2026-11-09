import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 4, 4, 7
    
    # We'll build U, V, W as lists of lists
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (28, rank)
    
    U_cols = []  # list of columns, each is a list of 16 entries
    V_cols = []  # list of columns, each is a list of 28 entries
    W_cols = []  # list of columns, each is a list of 28 entries
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries: dict {index: coeff}, etc."""
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        for idx, coeff in u_entries.items():
            u_col[idx] = coeff
        for idx, coeff in v_entries.items():
            v_col[idx] = coeff
        for idx, coeff in w_entries.items():
            w_col[idx] = coeff
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    def merge_add(d1, d2):
        """Merge two coefficient dicts by addition."""
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) + v
        return result
        
    def merge_sub(d1, d2):
        """Merge two coefficient dicts by subtraction."""
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) - v
        return result
        
    def scale(d, s):
        """Scale a coefficient dict."""
        return {k: v * s for k, v in d.items()}
        
    def merge_w(d1, d2, s1=1, s2=1):
        """Merge two C coefficient dicts with scaling."""
        result = {}
        for k, v in d1.items():
            result[k] = result.get(k, 0) + s1 * v
        for k, v in d2.items():
            result[k] = result.get(k, 0) + s2 * v
        return result

    def a_idx(i, j):
        return i * m + j
    
    def b_idx(j, k):
        return j * p + k
    
    def c_idx(i, k):
        return k * n + i
    
    def strassen_2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """Add 7 rank-1 terms for a 2x2 x 2x2 Strassen decomposition."""
        a00 = (a_rows[0], a_cols[0])
        a01 = (a_rows[0], a_cols[1])
        a10 = (a_rows[1], a_cols[0])
        a11 = (a_rows[1], a_cols[1])
        
        b00 = (b_rows[0], b_cols[0])
        b01 = (b_rows[0], b_cols[1])
        b10 = (b_rows[1], b_cols[0])
        b11 = (b_rows[1], b_cols[1])
        
        c00 = (c_rows[0], c_cols[0])
        c01 = (c_rows[0], c_cols[1])
        c10 = (c_rows[1], c_cols[0])
        c11 = (c_rows[1], c_cols[1])
        
        # M1 = (A00+A11)(B00+B11)
        add_term(
            {a_idx(*a00): 1, a_idx(*a11): 1},
            {b_idx(*b00): 1, b_idx(*b11): 1},
            {c_idx(*c00): 1, c_idx(*c11): 1}
        )
        # M2 = (A10+A11)B00
        add_term(
            {a_idx(*a10): 1, a_idx(*a11): 1},
            {b_idx(*b00): 1},
            {c_idx(*c10): 1, c_idx(*c11): -1}
        )
        # M3 = A00(B01-B11)
        add_term(
            {a_idx(*a00): 1},
            {b_idx(*b01): 1, b_idx(*b11): -1},
            {c_idx(*c01): 1, c_idx(*c11): 1}
        )
        # M4 = A11(B10-B00)
        add_term(
            {a_idx(*a11): 1},
            {b_idx(*b10): 1, b_idx(*b00): -1},
            {c_idx(*c00): 1, c_idx(*c10): 1}
        )
        # M5 = (A00+A01)B11
        add_term(
            {a_idx(*a00): 1, a_idx(*a01): 1},
            {b_idx(*b11): 1},
            {c_idx(*c00): -1, c_idx(*c01): 1}
        )
        # M6 = (A10-A00)(B00+B01)
        add_term(
            {a_idx(*a10): 1, a_idx(*a00): -1},
            {b_idx(*b00): 1, b_idx(*b01): 1},
            {c_idx(*c11): 1}
        )
        # M7 = (A01-A11)(B10+B11)
        add_term(
            {a_idx(*a01): 1, a_idx(*a11): -1},
            {b_idx(*b10): 1, b_idx(*b11): 1},
            {c_idx(*c00): 1}
        )
    
    def schoolbook_2x2_2x1(a_rows, a_cols, b_rows, b_col, c_rows, c_col):
        """Add 4 rank-1 terms for 2x2 x 2x1 schoolbook."""
        for i in range(2):
            for j in range(2):
                add_term(
                    {a_idx(a_rows[i], a_cols[j]): 1},
                    {b_idx(b_rows[j], b_col): 1},
                    {c_idx(c_rows[i], c_col): 1}
                )
    
    # Part 1: 4x4 x 4x4 (first 4 columns of B, cols 0-3)
    # Split into 2x2 blocks, apply Strassen at block level
    # A blocks: A11=(0,1)x(0,1), A12=(0,1)x(2,3), A21=(2,3)x(0,1), A22=(2,3)x(2,3)
    # B blocks: B11=(0,1)x(0,1), B12=(0,1)x(2,3), B21=(2,3)x(0,1), B22=(2,3)x(2,3)
    # C blocks: C11=(0,1)x(0,1), C12=(0,1)x(2,3), C21=(2,3)x(0,1), C22=(2,3)x(2,3)
    
    # Strassen at 4x4 level:
    # M1 = (A11+A22)(B11+B22) -> 2x2x2x2 Strassen for each... 
    # Actually, (A11+A22) is a 2x2 matrix, (B11+B22) is a 2x2 matrix.
    # Their product is a 2x2 matrix. We compute it with Strassen: 7 mults.
    # But wait, A11+A22 means we're adding the entries of A11 and A22.
    # So the "A" for this sub-problem has entries that are sums.
    
    # Hmm, this is where it gets tricky. In the tensor decomposition, we can't just add matrices and then multiply. We need to express each linear combination explicitly.
    
    # Let me reconsider. The Strassen algorithm at the 4x4 level:
    # M1 = (A11+A22)(B11+B22)
    # This is a 2x2 x 2x2 multiplication where the "A" matrix has entries:
    #   (A11+A22)[0][0] = A[0][0] + A[2][2]
    #   (A11+A22)[0][1] = A[0][1] + A[2][3]
    #   (A11+A22)[1][0] = A[1][0] + A[3][2]
    #   (A11+A22)[1][1] = A[1][1] + A[3][3]
    # And the "B" matrix has entries:
    #   (B11+B22)[0][0] = B[0][0] + B[2][2]
    #   (B11+B22)[0][1] = B[0][1] + B[2][3]
    #   (B11+B22)[1][0] = B[1][0] + B[3][2]
    #   (B11+B22)[1][1] = B[1][1] + B[3][3]
    # And the result contributes to C11 and C22:
    #   C11 += M1, C22 += M1 (with appropriate signs)
    
    # So for each Strassen sub-multiplication, the U, V, W entries involve sums/differences of A, B, C entries.
    
    # This means my strassen_2x2 function needs to be more general - it should take explicit linear combinations, not just single indices.
    
    # Let me redefine. Instead of passing indices, I'll pass the actual coefficient vectors.
    
    # Actually, let me just inline everything. It'll be verbose but correct.
    
    # Let me restart with a cleaner approach. I'll define the decomposition directly.
    
    # For the 4x4 x 4x4 part, I'll use the recursive Strassen.
    # The 7 block-level Strassen combinations are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 x 2x2 multiplication, done with Strassen (7 mults each).
    # Total: 7 * 7 = 49 mults.
    
    # For each Mi, the "A" and "B" are linear combinations of the original A and B entries.
    # I need to express each entry of these linear combinations as coefficients of the original entries.
    
    # Let me define a helper that, given a 2x2 "virtual" A matrix (as a 4-element list of coefficient dicts)
    # and a 2x2 "virtual" B matrix (similarly), and a 2x2 "virtual" C matrix (as coefficient dicts),
    # adds the 7 Strassen terms.
    
    def strassen_virtual(va, vb, vc):
        """
        va: 2x2 list of dicts {a_index: coeff} - the virtual A matrix
        vb: 2x2 list of dicts {b_index: coeff} - the virtual B matrix  
        vc: 2x2 list of dicts {c_index: coeff} - the virtual C matrix (how the result maps to C)
        """
        # va[i][j] is a dict of {a_idx: coeff} for virtual A[i][j]
        # vb[i][j] is a dict of {b_idx: coeff} for virtual B[i][j]
        # vc[i][j] is a dict of {c_idx: coeff} for virtual C[i][j]
        
        a00, a01 = va[0][0], va[0][1]
        a10, a11 = va[1][0], va[1][1]
        b00, b01 = vb[0][0], vb[0][1]
        b10, b11 = vb[1][0], vb[1][1]
        c00, c01 = vc[0][0], vc[0][1]
        c10, c11 = vc[1][0], vc[1][1]
        
        # M1 = (a00+a11)(b00+b11)
        add_term(
            merge_add(a00, a11),
            merge_add(b00, b11),
            merge_w(c00, c11, 1, 1)
        )
        # M2 = (a10+a11)b00
        add_term(
            merge_add(a10, a11),
            b00,
            merge_w(c10, c11, 1, -1)
        )
        # M3 = a00(b01-b11)
        add_term(
            a00,
            merge_sub(b01, b11),
            merge_w(c01, c11, 1, 1)
        )
        # M4 = a11(b10-b00)
        add_term(
            a11,
            merge_sub(b10, b00),
            merge_w(c00, c10, 1, 1)
        )
        # M5 = (a00+a01)b11
        add_term(
            merge_add(a00, a01),
            b11,
            merge_w(c00, c01, -1, 1)
        )
        # M6 = (a10-a00)(b00+b01)
        add_term(
            merge_sub(a10, a00),
            merge_add(b00, b01),
            c11
        )
        # M7 = (a01-a11)(b10+b11)
        add_term(
            merge_sub(a01, a11),
            merge_add(b10, b11),
            c00
        )
    
    # Now build the 4x4 x 4x4 part.
    # A blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B blocks (for first 4 cols):
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # C blocks:
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3
    
    def make_a_block(r1, r2, c1, c2):
        """Create a 2x2 virtual A block from actual A entries."""
        return [
            [{a_idx(r1, c1): 1}, {a_idx(r1, c2): 1}],
            [{a_idx(r2, c1): 1}, {a_idx(r2, c2): 1}]
        ]
    
    def make_b_block(r1, r2, c1, c2):
        """Create a 2x2 virtual B block from actual B entries."""
        return [
            [{b_idx(r1, c1): 1}, {b_idx(r1, c2): 1}],
            [{b_idx(r2, c1): 1}, {b_idx(r2, c2): 1}]
        ]
    
    def make_c_block(r1, r2, c1, c2):
        """Create a 2x2 virtual C block from actual C entries."""
        return [
            [{c_idx(r1, c1): 1}, {c_idx(r1, c2): 1}],
            [{c_idx(r2, c1): 1}, {c_idx(r2, c2): 1}]
        ]
    
    A11 = make_a_block(0, 1, 0, 1)
    A12 = make_a_block(0, 1, 2, 3)
    A21 = make_a_block(2, 3, 0, 1)
    A22 = make_a_block(2, 3, 2, 3)
    
    B11 = make_b_block(0, 1, 0, 1)
    B12 = make_b_block(0, 1, 2, 3)
    B21 = make_b_block(2, 3, 0, 1)
    B22 = make_b_block(2, 3, 2, 3)
    
    C11 = make_c_block(0, 1, 0, 1)
    C12 = make_c_block(0, 1, 2, 3)
    C21 = make_c_block(2, 3, 0, 1)
    C22 = make_c_block(2, 3, 2, 3)
    
    def block_add(b1, b2):
        """Add two 2x2 virtual blocks."""
        return [
            [merge_add(b1[0][0], b2[0][0]), merge_add(b1[0][1], b2[0][1])],
            [merge_add(b1[1][0], b2[1][0]), merge_add(b1[1][1], b2[1][1])]
        ]
    def block_sub(b1, b2):
        """Subtract two 2x2 virtual blocks."""
        return [
            [merge_sub(b1[0][0], b2[0][0]), merge_sub(b1[0][1], b2[0][1])],
            [merge_sub(b1[1][0], b2[1][0]), merge_sub(b1[1][1], b2[1][1])]
        ]
    
    # C mapping for each Strassen combination:
    # M1 -> C11 (+1), C22 (+1)
    # M2 -> C21 (+1), C22 (-1)
    # M3 -> C12 (+1), C22 (+1)
    # M4 -> C11 (+1), C21 (+1)
    # M5 -> C11 (-1), C12 (+1)
    # M6 -> C22 (+1)
    # M7 -> C11 (+1)
    
    def c_map_m1():
        return block_add(C11, C22)  # C11 + C22
    def c_map_m2():
        return [
            [merge_w(C21[0][0], C22[0][0], 1, -1), merge_w(C21[0][1], C22[0][1], 1, -1)],
            [merge_w(C21[1][0], C22[1][0], 1, -1), merge_w(C21[1][1], C22[1][1], 1, -1)]
        ]
    def c_map_m3():
        return [
            [merge_w(C12[0][0], C22[0][0], 1, 1), merge_w(C12[0][1], C22[0][1], 1, 1)],
            [merge_w(C12[1][0], C22[1][0], 1, 1), merge_w(C12[1][1], C22[1][1], 1, 1)]
        ]
    def c_map_m4():
        return [
            [merge_w(C11[0][0], C21[0][0], 1, 1), merge_w(C11[0][1], C21[0][1], 1, 1)],
            [merge_w(C11[1][0], C21[1][0], 1, 1), merge_w(C11[1][1], C21[1][1], 1, 1)]
        ]
    def c_map_m5():
        return [
            [merge_w(C11[0][0], C12[0][0], -1, 1), merge_w(C11[0][1], C12[0][1], -1, 1)],
            [merge_w(C11[1][0], C12[1][0], -1, 1), merge_w(C11[1][1], C12[1][1], -1, 1)]
        ]
    def c_map_m6():
        return C22
    def c_map_m7():
        return C11
    
    # Now apply Strassen at the 4x4 level
    strassen_virtual(block_add(A11, A22), block_add(B11, B22), c_map_m1())
    strassen_virtual(block_add(A21, A22), B11, c_map_m2())
    strassen_virtual(A11, block_sub(B12, B22), c_map_m3())
    strassen_virtual(A22, block_sub(B21, B11), c_map_m4())
    strassen_virtual(block_add(A11, A12), B22, c_map_m5())
    strassen_virtual(block_sub(A21, A11), block_add(B11, B12), c_map_m6())
    strassen_virtual(block_sub(A12, A22), block_add(B21, B22), c_map_m7())
    
    # Part 2: 4x4 x 4x3 (last 3 columns of B, cols 4-6)
    # Split B into B_left (4x2, cols 4-5) and B_right (4x1, col 6)
    
    # For 4x4 x 4x2:
    # Split A into 2x2 blocks, B_left into [2x2; 2x2]
    # C_left (4x2) split into [C1_left (2x2); C2_left (2x2)]
    # C1_left = [A11 A12] * [BL11; BL21] where BL11=B[0:2,4:6], BL21=B[2:4,4:6]
    # C2_left = [A21 A22] * [BL11; BL21]
    
    # Each is 2x4 x 4x2. Split the 4 in the middle:
    # C1_left = A11*BL11 + A12*BL21 (two 2x2 x 2x2 multiplications)
    # Use Strassen for each: 7+7 = 14
    
    # Similarly for C2_left: 14
    
    # For 4x4 x 4x1:
    # C1_right = [A11 A12] * [BR1; BR2] where BR1=B[0:2,6], BR2=B[2:4,6]
    # C2_right = [A21 A22] * [BR1; BR2]
    # Each is 2x4 x 4x1 = A11*BR1 + A12*BR2 (two 2x2 x 2x1)
    # Schoolbook for each 2x2 x 2x1: 4 mults
    # Total: 4+4 = 8 per C block, 16 total
    
    # 4x2 part:
    BL11 = make_b_block(0, 1, 4, 5)
    BL21 = make_b_block(2, 3, 4, 5)
    C1L = make_c_block(0, 1, 4, 5)
    C2L = make_c_block(2, 3, 4, 5)
    
    # C1L = A11*BL11 + A12*BL21
    # This is a 2x4 x 4x2 product. Split:
    # = [A11, A12] * [BL11; BL21]
    # = A11*BL11 + A12*BL21
    # Compute A11*BL11 with Strassen, A12*BL21 with Strassen
    strassen_virtual(A11, BL11, C1L)
    strassen_virtual(A12, BL21, C1L)
    strassen_virtual(A21, BL11, C2L)
    strassen_virtual(A22, BL21, C2L)
    
    # 4x1 part:
    BR1 = make_b_block(0, 1, 6, 6)  # This is 2x1, not 2x2!
    # Oops, I need a different function for 2x1 blocks.
    
    # Let me handle the 2x2 x 2x1 case separately.
    
    # For 2x2 x 2x1 schoolbook:
    # A is 2x2, B is 2x1, C is 2x1
    # C[0] = A[0][0]*B[0] + A[0][1]*B[1]
    # C[1] = A[1][0]*B[0] + A[1][1]*B[1]
    # 4 rank-1 terms
    
    def schoolbook_2x2_2x1(va, vb, vc):
        """
        va: 2x2 virtual A block
        vb: 2x1 virtual B block (list of 2 dicts)
        vc: 2x1 virtual C block (list of 2 dicts)
        """
        for i in range(2):
            for j in range(2):
                add_term(
                    va[i][j],
                    vb[j],
                    vc[i]
                )
    
    BR_top = [{b_idx(0, 6): 1}, {b_idx(1, 6): 1}]  # B[0:2, 6]
    BR_bot = [{b_idx(2, 6): 1}, {b_idx(3, 6): 1}]  # B[2:4, 6]
    CR_top = [{c_idx(0, 6): 1}, {c_idx(1, 6): 1}]  # C[0:2, 6]
    CR_bot = [{c_idx(2, 6): 1}, {c_idx(3, 6): 1}]  # C[2:4, 6]
    
    # C1_right = A11*BR_top + A12*BR_bot
    schoolbook_2x2_2x1(A11, BR_top, CR_top)
    schoolbook_2x2_2x1(A12, BR_bot, CR_top)
    schoolbook_2x2_2x1(A21, BR_top, CR_bot)
    schoolbook_2x2_2x1(A22, BR_bot, CR_bot)
    
    # Convert to final format
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    # Clean up: convert to int where possible
    def clean_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if isinstance(val, float) and val == int(val):
                    new_row.append(int(val))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    U = clean_matrix(U)
    V = clean_matrix(V)
    W = clean_matrix(W)
    
    return {"U": U, "V": V, "W": W}
