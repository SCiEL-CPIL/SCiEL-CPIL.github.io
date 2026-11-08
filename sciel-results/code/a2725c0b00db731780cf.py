"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 77 decomposition
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2 Strassen decomposition
    # u_idx: list of 4 global indices for U (i*m+j)
    # v_idx: list of 4 global indices for V (j*p+k)
    # w_idx: list of 4 global indices for W (k*n+i)
    # coeffs_u, coeffs_v, coeffs_w: 7x4 matrices of coefficients for the 7 Strassen terms
    # This is too generic. I will hardcode the 7 terms for 2x2 Strassen.
    
    # Strassen 2x2 terms (u, v, w)
    # u, v, w are vectors of length 4 (flattened 2x2)
    # Term 1: (A00+A11)(B00+B11) -> C00, C11
    # Term 2: (A10+A11)B00 -> C10
    # Term 3: A00(B01-B11) -> C01
    # Term 4: A11(B10-B00) -> C10, C11
    # Term 5: (A00+A01)B11 -> C01, C11
    # Term 6: (A10-A00)(B00+B01) -> C10
    # Term 7: (A01-A11)(B10+B11) -> C01, C11
    
    # We need to map local 2x2 indices to global indices.
    
    def add_2x2_strassen(u_map, v_map, w_map):
        nonlocal col
        # u_map[i,j] = global U index
        # v_map[i,j] = global V index
        # w_map[i,j] = global W index
        
        # 7 terms
        # Each term is a triple (u_vec, v_vec, w_vec)
        # u_vec, v_vec, w_vec are length 4 lists [val00, val01, val10, val11]
        
        terms = [
            # 1
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            # 2
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            # 3
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
            # 4
            ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, 1]),
            # 5
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),
            # 6
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 0]),
            # 7
            ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]),
        ]
        
        for u_vals, v_vals, w_vals in terms:
            for local_idx in range(4):
                if u_vals[local_idx] != 0:
                    i, j = divmod(local_idx, 2)
                    U[u_map[i][j]][col] += u_vals[local_idx]
                if v_vals[local_idx] != 0:
                    i, j = divmod(local_idx, 2)
                    V[v_map[i][j]][col] += v_vals[local_idx]
                if w_vals[local_idx] != 0:
                    i, j = divmod(local_idx, 2)
                    W[w_map[i][j]][col] += w_vals[local_idx]
            col += 1

    # Part 1: T_4,4,4 (First 4 cols of B)
    # Blocks:
    # A: [[A00, A01], [A10, A11]]
    # B: [[B00, B01], [B10, B11]]
    
    # Strassen Steps for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Ml is a 2x2 multiplication.
    # We need to call add_2x2_strassen for each Ml, mapping the local 2x2 indices to the global 4x4 indices.
    
    # Mapping for M1: A_part = A00+A11, B_part = B00+B11
    # The "A" for this 2x2 mult is the sum of two 2x2 blocks.
    # This means the u_map for this call must reflect that local (0,0) corresponds to global A00(0,0) AND A11(0,0)?
    # No, the linear combination is on the inputs.
    # The rank-1 tensor is u ⊗ v ⊗ w.
    # If u = u_A00 + u_A11, then U entries are sum of entries from A00 and A11.
    
    # My add_2x2_strassen function assumes u_map points to specific global indices.
    # I need to modify it to accept linear combinations.
    
    # Revised plan:
    # Generate the 49 terms explicitly.
    
    # I will create a list of 49 tuples (u_coeffs, v_coeffs, w_coeffs)
    # where u_coeffs is a dict {global_index: coeff}
    
    # This is getting long. I will use a simpler approach:
    # Just use the schoolbook for the 4x4x4 part? No, rank 64.
    # 64 + 28 = 92. Still better than 96.
    # But 49 + 28 = 77 is better.
    
    # I will implement the 77 rank decomposition by explicitly writing out the 77 terms.
    
    # Due to complexity of hardcoding 77 terms, I will use the schoolbook for T_4,4,4 (rank 64)
    # and the Strassen-derived rank 28 for T_4,4,2.
    # Total 92.
    
    # Wait, 92 < 96. This is a valid improvement.
    # Let's stick to this to ensure correctness and simplicity.
    
    # Reset col
    col = 0
    rank = 92
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: T_4,4,4 Schoolbook (Rank 64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * 4 + j][col] = 1
                V[j * 6 + k][col] = 1 # B is 4x6, so j*p+k
                W[k * 4 + i][col] = 1 # C is 4x4 (first 4 cols), k*n+i
                col += 1
                
    # Part 2: T_4,4,2 Strassen (Rank 28)
    # B cols 4,5. C cols 4,5.
    # 4 independent 2x2 multiplications.
    
    # Mapping for 2x2 blocks in 4x4 A and 4x2 B
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,4), B01 (0,5), B10 (1,4), B11 (1,5) -- Wait, B is 4x2.
    # B rows 0,1 -> B_top. Rows 2,3 -> B_bot.
    # B_top: cols 4,5. B_bot: cols 4,5.
    
    # 1. A00 * B_top
    # A00: i in 0,1; j in 0,1.
    # B_top: j in 0,1; k in 4,5.
    # C_top: k in 4,5; i in 0,1.
    
    def add_2x2_block(i_off, j_off, k_off, b_row_off):
        nonlocal col
        # i_off, j_off: offset for A block (2x2)
        # k_off: offset for C cols (0 or 1 within the 2 cols)
        # b_row_off: offset for B rows (0 or 2)
        
        # Use Strassen for this 2x2 multiplication
        # Local indices 0,1 for i,j,k
        
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
            ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, 1]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 0]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]),
        ]
        
        for u_vals, v_vals, w_vals in terms:
            for local_idx in range(4):
                if u_vals[local_idx] != 0:
                    li, lj = divmod(local_idx, 2)
                    gi = i_off + li
                    gj = j_off + lj
                    U[gi * 4 + gj][col] += u_vals[local_idx]
                if v_vals[local_idx] != 0:
                    li, lj = divmod(local_idx, 2)
                    gj = b_row_off + li # B row
                    gk = 4 + k_off + lj # B col (4,5)
                    V[gj * 6 + gk][col] += v_vals[local_idx]
                if w_vals[local_idx] != 0:
                    li, lj = divmod(local_idx, 2)
                    gk = 4 + k_off + li # C col
                    gi = i_off + lj # C row
                    W[gk * 4 + gi][col] += w_vals[local_idx]
            col += 1
            
    # 4 blocks:
    # A00 (0,0) * B_top (0, 4,5) -> C_top (0, 4,5)
    add_2x2_block(0, 0, 0, 0)
    # A01 (0,1) * B_bot (2, 4,5) -> C_top (0, 4,5)
    # Wait, C_top = A00*B_top + A01*B_bot
    # My add_2x2_block adds to W.
    # I need to ensure the W indices are correct.
    # C_top rows 0,1. C_bot rows 2,3.
    
    # 1. A00 * B_top -> C_top
    add_2x2_block(0, 0, 0, 0)
    # 2. A01 * B_bot -> C_top
    # A01 is i in 0,1; j in 2,3.
    # B_bot is j in 2,3; k in 4,5.
    # C_top is k in 4,5; i in 0,1.
    # My function add_2x2_block(i_off, j_off, k_off, b_row_off)
    # i_off=0, j_off=2, k_off=0, b_row_off=2
    add_2x2_block(0, 2, 0, 2)
    
    # 3. A10 * B_top -> C_bot
    # A10 is i in 2,3; j in 0,1.
    # B_top is j in 0,1; k in 4,5.
    # C_bot is k in 4,5; i in 2,3.
    add_2x2_block(2, 0, 0, 0)
    
    # 4. A11 * B_bot -> C_bot
    # A11 is i in 2,3; j in 2,3.
    # B_bot is j in 2,3; k in 4,5.
    # C_bot is k in 4,5; i in 2,3.
    add_2x2_block(2, 2, 0, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
