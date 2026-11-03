"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (32, rank)  
    # W: (p*n, rank) = (32, rank)
    
    # Strassen decomposition
    # A is 4x4, split into 2x2 blocks of 2x2
    # B is 4x8, split into 2x2 blocks of 2x4
    # C is 4x8, split into 2x2 blocks of 2x4
    
    # Each Strassen term Mi = Xi * Yi where Xi is 2x2 (linear combo of A blocks)
    # and Yi is 2x4 (linear combo of B blocks)
    # Schoolbook for 2x2 * 2x4: 16 scalar products
    
    terms = []
    
    # Define the 7 Strassen combinations
    # Each entry: (A_block_combo, B_block_combo)
    # A blocks: (row_block, col_block) where row_block in {0,1}, col_block in {0,1}
    # B blocks: (row_block, col_block) where row_block in {0,1}, col_block in {0,1}
    # col_block for B: 0 means cols 0-3, 1 means cols 4-7
    
    # M1 = (A11+A22) * (B11+B21)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B21-B22)
    # M4 = A22 * (B11-B12)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B12+B22)
    # M7 = (A12-A21) * (B11+B21)
    
    # For each term, we have a linear combination of A blocks and B blocks
    # A block (rb, cb) corresponds to A[rb*2:(rb+1)*2, cb*2:(cb+1)*2]
    # B block (rb, cb) corresponds to B[rb*2:(rb+1)*2, cb*4:(cb+1)*4]
    
    strassen_terms = [
        # (A_coeff_dict, B_coeff_dict)
        # A_coeff: {(rb,cb): coeff}
        # B_coeff: {(rb,cb): coeff}
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,0): 1}),  # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),              # M2
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}),             # M3
        ({(1,1): 1}, {(0,0): 1, (0,1): -1}),             # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),              # M5
        ({(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}),   # M6
        ({(0,1): 1, (1,0): -1}, {(0,0): 1, (1,0): 1}),   # M7
    ]
    
    rank = 0
    U = []
    V = []
    W = []
    
    for a_coeffs, b_coeffs in strassen_terms:
        # For each scalar product in the 2x2 * 2x4 schoolbook
        # The 2x2 X has entries X[i,j] for i in {0,1}, j in {0,1}
        # The 2x4 Y has entries Y[j,k] for j in {0,1}, k in {0,1,2,3}
        # Result Z[i,k] = sum_j X[i,j] * Y[j,k]
        
        for i in range(2):  # row of X
            for j in range(2):  # col of X / row of Y
                for k in range(4):  # col of Y
                    # This scalar product contributes X[i,j] * Y[j,k] to Z[i,k]
                    
                    # X[i,j] is a linear combination of A blocks
                    # Y[j,k] is a linear combination of B blocks
                    # Z[i,k] contributes to C at position determined by Strassen output
                    
                    col = rank
                    u_row_entries = [0.0] * 16  # will fill
                    v_row_entries = [0.0] * 32
                    w_row_entries = [0.0] * 32
                    
                    # Determine which C block this contributes to
                    # We'll handle the C-block assignment after
                    rank += 1
    
    # This approach is getting complicated. Let me restructure.
    
    # Better approach: build U, V, W directly.
    # Each rank-1 term corresponds to a scalar product x * y where:
    # x = sum over A blocks of coeff * A[local_i, local_j]
    # y = sum over B blocks of coeff * B[local_j, local_k]
    # and it contributes to C[global_i, global_k] with some coefficient.
    
    U = []  # will be list of lists, shape (16, rank)
    V = []  # shape (32, rank)
    W = []  # shape (32, rank)
    
    term_list = []  # each term: (u_vec, v_vec, w_vec) where u_vec has 16 entries, v_vec 32, w_vec 32
    
    for a_coeffs, b_coeffs in strassen_terms:
        for i_local in range(2):
            for j_local in range(2):
                for k_local in range(4):
                    u_vec = [0] * 16
                    v_vec = [0] * 32
                    w_vec = [0] * 32
                    
                    # X[i_local, j_local] = sum of A block contributions
                    for (rb, cb), coeff in a_coeffs.items():
                        # A block (rb, cb) is at A[rb*2 + i_local, cb*2 + j_local]
                        # U index: (rb*2 + i_local) * m + (cb*2 + j_local) = (rb*2+i_local)*4 + (cb*2+j_local)
                        u_idx = (rb * 2 + i_local) * 4 + (cb * 2 + j_local)
                        u_vec[u_idx] += coeff
                    
                    # Y[j_local, k_local] = sum of B block contributions
                    for (rb, cb), coeff in b_coeffs.items():
                        # B block (rb, cb) is at B[rb*2 + j_local, cb*4 + k_local]
                        # V index: (rb*2 + j_local) * p + (cb*4 + k_local) = (rb*2+j_local)*8 + (cb*4+k_local)
                        v_idx = (rb * 2 + j_local) * 8 + (cb * 4 + k_local)
                        v_vec[v_idx] += coeff
                    
                    # Now determine where this contributes in C
                    # The Strassen term with these A and B block combos contributes to specific C blocks
                    # We need to figure out the C block and coefficient
                    # This is complex - let me use a different approach
                    
                    term_list.append((u_vec, v_vec, w_vec, None))  # None for now
    
    # This is getting too complex. Let me use a cleaner formulation.
    
    # Actually, let me just directly enumerate all 112 terms.
    
    # The 7 Strassen terms, each is a 2x2 * 2x4 product (16 scalar products)
    # For each scalar product, I need to determine:
    # 1. Which A entries are involved (for U)
    # 2. Which B entries are involved (for V)  
    # 3. Which C entry it contributes to and with what coefficient (for W)
    
    # Let me define the Strassen terms more explicitly.
    
    # I'll build the full decomposition by tracking contributions.
    
    # Clear and restart
    U = [[0] * 112 for _ in range(16)]
    V = [[0] * 112 for _ in range(32)]
    W = [[0] * 112 for _ in range(32)]
    
    col = 0
    
    # For each of the 7 Strassen terms:
    # Term t has A-side: linear combo of 2x2 A blocks
    #            B-side: linear combo of 2x4 B blocks
    # The product is a 2x4 matrix Z where Z[i,k] = sum_j X[i,j]*Y[j,k]
    # Each Z[i,k] contributes to a specific C block with a coefficient.
    
    # Let me define the mapping from (Strassen term, i, k) to C block and coefficient.
    
    # M1 = (A11+A22)*(B11+B21) -> contributes to C11 (+1), C22 (+1)
    # M2 = (A21+A22)*B11 -> contributes to C21 (+1), C22 (-1)
    # M3 = A11*(B21-B22) -> contributes to C12 (+1), C22 (+1)
    # M4 = A22*(B11-B12) -> contributes to C11 (+1), C21 (+1)
    # M5 = (A11+A12)*B22 -> contributes to C11 (-1), C12 (+1)
    # M6 = (A21-A11)*(B12+B22) -> contributes to C22 (+1)
    # M7 = (A12-A21)*(B11+B21) -> contributes to C11 (+1)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen term t, the 2x4 result Z_t has entries Z_t[i,k] for i in {0,1}, k in {0,1,2,3}
    # Z_t[i,k] contributes to:
    #   C11[i, k] with coeff c11[t]
    #   C12[i, k] with coeff c12[t]
    #   C21[i, k] with coeff c21[t]
    #   C22[i, k] with coeff c22[t]
    
    c_coeffs = {
        0: {'c11': 1, 'c12': 0, 'c21': 0, 'c22': 1},   # M1
        1: {'c11': 0, 'c12': 0, 'c21': 1, 'c22': -1},   # M2
        2: {'c11': 0, 'c12': 1, 'c21': 0, 'c22': 1},    # M3
        3: {'c11': 1, 'c12': 0, 'c21': 1, 'c22': 0},    # M4
        4: {'c11': -1, 'c12': 1, 'c21': 0, 'c22': 0},   # M5
        5: {'c11': 0, 'c12': 0, 'c21': 0, 'c22': 1},    # M6
        6: {'c11': 1, 'c12': 0, 'c21': 0, 'c22': 0},    # M7
    }
    
    # A-side coefficients for each Strassen term
    # a_coeffs[t] = {(rb, cb): coeff}
    a_coeffs = [
        {(0,0): 1, (1,1): 1},      # M1: A11+A22
        {(1,0): 1, (1,1): 1},      # M2: A21+A22
        {(0,0): 1},                 # M3: A11
        {(1,1): 1},                 # M4: A22
        {(0,0): 1, (0,1): 1},      # M5: A11+A12
        {(1,0): 1, (0,0): -1},     # M6: A21-A11
        {(0,1): 1, (1,0): -1},     # M7: A12-A21
    ]
    
    # B-side coefficients for each Strassen term
    # b_coeffs[t] = {(rb, cb): coeff}
    b_coeffs = [
        {(0,0): 1, (1,0): 1},      # M1: B11+B21
        {(0,0): 1},                 # M2: B11
        {(1,0): 1, (1,1): -1},     # M3: B21-B22
        {(0,0): 1, (0,1): -1},     # M4: B11-B12
        {(1,1): 1},                 # M5: B22
        {(0,1): 1, (1,1): 1},      # M6: B12+B22
        {(0,0): 1, (1,0): 1},      # M7: B11+B21
    ]
    
    for t in range(7):
        for i in range(2):  # row within the 2x2 X
            for j in range(2):  # col within X / row within Y
                for k in range(4):  # col within Y
                    # This scalar product: X[i,j] * Y[j,k]
                    
                    # U contributions: X[i,j] is linear combo of A entries
                    for (rb, cb), coeff in a_coeffs[t].items():
                        a_row = rb * 2 + i
                        a_col = cb * 2 + j
                        u_idx = a_row * 4 + a_col  # U has shape (16, rank), indexed by a_row*4+a_col
                        U[u_idx][col] += coeff
                    
                    # V contributions: Y[j,k] is linear combo of B entries
                    for (rb, cb), coeff in b_coeffs[t].items():
                        b_row = rb * 2 + j
                        b_col = cb * 4 + k
                        v_idx = b_row * 8 + b_col  # V has shape (32, rank), indexed by b_row*8+b_col
                        V[v_idx][col] += coeff
                    
                    # W contributions: this contributes to C blocks
                    # Z[i,k] where i is row within top/bottom half, k is col within left/right 4
                    cc = c_coeffs[t]
                    if cc['c11'] != 0:
                        c_row = i  # top half
                        c_col = k  # left 4 cols
                        w_idx = c_col * 4 + c_row  # W[k*n+i] = W[c_col*4+c_row]
                        W[w_idx][col] += cc['c11']
                    if cc['c12'] != 0:
                        c_row = i  # top half
                        c_col = 4 + k  # right 4 cols
                        w_idx = c_col * 4 + c_row
                        W[w_idx][col] += cc['c12']
                    if cc['c21'] != 0:
                        c_row = 2 + i  # bottom half
                        c_col = k  # left 4 cols
                        w_idx = c_col * 4 + c_row
                        W[w_idx][col] += cc['c21']
                    if cc['c22'] != 0:
                        c_row = 2 + i  # bottom half
                        c_col = 4 + k  # right 4 cols
                        w_idx = c_col * 4 + c_row
                        W[w_idx][col] += cc['c22']
                    
                    col += 1
    
    assert col == 112, f"Expected 112, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
