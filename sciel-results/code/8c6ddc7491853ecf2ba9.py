"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Construct a rank-42 decomposition using a block-based Strassen approach.
    # We split the 4x6 B matrix into three blocks of 2 columns each.
    # For each block, we compute C_block = A * B_block where A is 2x4 and B_block is 4x2.
    # We split A into A1 (cols 0,1) and A2 (cols 2,3), and B_block into B_top (rows 0,1) and B_bottom (rows 2,3).
    # Then C_block = A1 * B_top + A2 * B_bottom.
    # Each product is a 2x2 * 2x2 multiplication, which we handle with Strassen (rank 7).
    # Total rank = 3 blocks * 2 products * 7 Strassen multiplications = 42.
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    rank = 42
    
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen formulas for 2x2 matrix multiplication:
    # M1 = X00 * (Y00 + Y11)
    # M2 = (X00 + X10) * Y11
    # M3 = (X00 + X01) * Y00
    # M4 = X11 * (Y10 + Y00)
    # M5 = (X00 + X11) * (Y01 + Y10)
    # M6 = (X00 - X11) * (Y00 + Y01)
    # M7 = (X10 - X00) * (Y11 + Y01)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # We'll define the Strassen coefficients as a list of (U_coeffs, V_coeffs, W_coeffs)
    # where each is a dict mapping local indices to coefficients.
    
    # Local indices for 2x2 matrix:
    # X (A part): (0,0), (0,1), (1,0), (1,1) -> flat indices 0,1,2,3
    # Y (B part): (0,0), (0,1), (1,0), (1,1) -> flat indices 0,1,2,3
    # Z (C part): (0,0), (0,1), (1,0), (1,1) -> flat indices 0,1,2,3
    
    # For each Strassen multiplication, we have 7 terms.
    # Each term has:
    # - U coefficients for the X part (2x2)
    # - V coefficients for the Y part (2x2)
    # - W coefficients for the Z part (2x2)
    
    # Let's define the 7 Strassen terms for a single 2x2x2 multiplication.
    # Term 0: M1 = X00 * (Y00 + Y11)
    # Term 1: M2 = (X00 + X10) * Y11
    # Term 2: M3 = (X00 + X01) * Y00
    # Term 3: M4 = X11 * (Y10 + Y00)
    # Term 4: M5 = (X00 + X11) * (Y01 + Y10)
    # Term 5: M6 = (X00 - X11) * (Y00 + Y01)
    # Term 6: M7 = (X10 - X00) * (Y11 + Y01)
    
    # Strassen terms as (U_local, V_local, W_local)
    # U_local: dict of (i,j) -> coeff for X
    # V_local: dict of (j,k) -> coeff for Y
    # W_local: dict of (i,k) -> coeff for Z
    
    # For each term, we need to express the product as sum of outer products.
    # Actually, each Strassen term is a single multiplication of two linear combinations.
    # So U has one non-zero entry (or a few if it's a sum), V has one non-zero entry, W has contributions.
    
    # Let me think more carefully. In the tensor decomposition, each rank-1 term is:
    # U_r[i,j] * V_r[j,k] * W_r[k,i] summed over r.
    # For Strassen, each "multiplication" M_s = (linear combo of X) * (linear combo of Y)
    # contributes to the output Z.
    
    # So for each Strassen term s (0..6):
    # - U_s has non-zero entries where the linear combo of X is non-zero
    # - V_s has non-zero entries where the linear combo of Y is non-zero
    # - W_s has non-zero entries where Z receives contribution from M_s
    
    # Let me define the linear combinations:
    # M1: X00 * (Y00 + Y11)
    #   U: {(0,0): 1}
    #   V: {(0,0): 1, (1,1): 1}
    #   W: {(0,0): 1, (1,1): 1}  (since M1 contributes to Z00 and Z11)
    #   Wait, Z00 = M1 + M4 - M5 + M7, Z11 = M1 - M2 + M3 + M6
    #   So M1 contributes +1 to Z00 and +1 to Z11.
    
    # Let me list all 7 terms with their U, V, W coefficients:
    
    # Term 0 (M1): X00 * (Y00 + Y11)
    #   U: {(0,0): 1}
    #   V: {(0,0): 1, (1,1): 1}
    #   W: {(0,0): 1, (1,1): 1}
    
    # Term 1 (M2): (X00 + X10) * Y11
    #   U: {(0,0): 1, (1,0): 1}
    #   V: {(1,1): 1}
    #   W: {(1,0): 1, (1,1): -1}  (M2 contributes +1 to Z10, -1 to Z11)
    
    # Term 2 (M3): (X00 + X01) * Y00
    #   U: {(0,0): 1, (0,1): 1}
    #   V: {(0,0): 1}
    #   W: {(0,1): 1, (1,1): 1}  (M3 contributes +1 to Z01, +1 to Z11)
    
    # Term 3 (M4): X11 * (Y10 + Y00)
    #   U: {(1,1): 1}
    #   V: {(1,0): 1, (0,0): 1}
    #   W: {(0,0): 1, (1,0): 1}  (M4 contributes +1 to Z00, +1 to Z10)
    
    # Term 4 (M5): (X00 + X11) * (Y01 + Y10)
    #   U: {(0,0): 1, (1,1): 1}
    #   V: {(0,1): 1, (1,0): 1}
    #   W: {(0,0): -1, (0,1): 1}  (M5 contributes -1 to Z00, +1 to Z01)
    
    # Term 5 (M6): (X00 - X11) * (Y00 + Y01)
    #   U: {(0,0): 1, (1,1): -1}
    #   V: {(0,0): 1, (0,1): 1}
    #   W: {(1,1): 1}  (M6 contributes +1 to Z11)
    
    # Term 6 (M7): (X10 - X00) * (Y11 + Y01)
    #   U: {(1,0): 1, (0,0): -1}
    #   V: {(1,1): 1, (0,1): 1}
    #   W: {(0,0): 1}  (M7 contributes +1 to Z00)
    
    # Now, for each block b (0,1,2) and each sub-product (top or bottom), we have a Strassen instance.
    # Block b corresponds to B columns 2b, 2b+1 and C columns 2b, 2b+1.
    # Sub-product "top": A1 (A cols 0,1) * B_top (B rows 0,1, cols 2b,2b+1)
    # Sub-product "bottom": A2 (A cols 2,3) * B_bottom (B rows 2,3, cols 2b,2b+1)
    
    # Let's define the mapping from local (i,j) to global indices.
    # For A part (X):
    #   Top sub-product: i in {0,1}, j in {0,1} -> global A index = i*m + j = i*4 + j
    #   Bottom sub-product: i in {0,1}, j in {2,3} -> global A index = i*m + j = i*4 + j
    # For B part (Y):
    #   Top sub-product: j in {0,1}, k in {2b, 2b+1} -> global B index = j*p + k = j*6 + k
    #   Bottom sub-product: j in {2,3}, k in {2b, 2b+1} -> global B index = j*p + k = j*6 + k
    # For C part (Z):
    #   i in {0,1}, k in {2b, 2b+1} -> global C index = k*n + i = k*2 + i (trace convention)
    
    # Let's build the decomposition.
    # We'll iterate over blocks b, sub-products (top/bottom), and Strassen terms.
    
    idx = 0
    for b in range(3):  # 3 blocks of 2 columns
        for sub in range(2):  # 0=top, 1=bottom
            # Determine the global index mappings
            if sub == 0:  # top: A cols 0,1, B rows 0,1
                a_j_offset = 0  # j in {0,1} -> global j = j + 0
                b_j_offset = 0  # j in {0,1} -> global j = j + 0
            else:  # bottom: A cols 2,3, B rows 2,3
                a_j_offset = 2  # j in {0,1} -> global j = j + 2
                b_j_offset = 2  # j in {0,1} -> global j = j + 2
            
            # B and C column offset
            k_offset = 2 * b  # k in {0,1} -> global k = k + k_offset
            
            # Define the Strassen terms
            strassen_terms = [
                # (U_local, V_local, W_local)
                ({(0,0): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # M1
                ({(0,0): 1, (1,0): 1}, {(1,1): 1}, {(1,0): 1, (1,1): -1}),  # M2
                ({(0,0): 1, (0,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): 1}),  # M3
                ({(1,1): 1}, {(1,0): 1, (0,0): 1}, {(0,0): 1, (1,0): 1}),  # M4
                ({(0,0): 1, (1,1): 1}, {(0,1): 1, (1,0): 1}, {(0,0): -1, (0,1): 1}),  # M5
                ({(0,0): 1, (1,1): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),  # M6
                ({(1,0): 1, (0,0): -1}, {(1,1): 1, (0,1): 1}, {(0,0): 1}),  # M7
            ]
            
            for U_local, V_local, W_local in strassen_terms:
                # Fill U
                for (li, lj), coeff in U_local.items():
                    global_i = li
                    global_j = lj + a_j_offset
                    u_idx = global_i * m + global_j
                    U[u_idx][idx] = coeff
                
                # Fill V
                for (lj, lk), coeff in V_local.items():
                    global_j = lj + b_j_offset
                    global_k = lk + k_offset
                    v_idx = global_j * p + global_k
                    V[v_idx][idx] = coeff
                
                # Fill W
                for (li, lk), coeff in W_local.items():
                    global_i = li
                    global_k = lk + k_offset
                    w_idx = global_k * n + global_i
                    W[w_idx][idx] = coeff
                
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
