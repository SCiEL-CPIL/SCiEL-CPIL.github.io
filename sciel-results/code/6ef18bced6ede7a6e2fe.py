"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * 7 = 28
    rank = 28
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # U_strassen[i,j,r], V_strassen[j,k,r], W_strassen[k,i,r]
    # i,j,k in {0,1}
    # r in 0..6
    
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # Coefficients for U (A)
    # U[i, j, r]
    U_s = [
        [[0, 0, 1, 0, 1, 0, 0],   # i=0, j=0 (a00)
         [0, 0, 0, 0, 1, -1, 1]],  # i=0, j=1 (a01)
        [[0, 1, 0, 0, 0, 1, 0],   # i=1, j=0 (a10)
         [1, 1, 0, 1, 0, 0, -1]]  # i=1, j=1 (a11)
    ]
    
    # Coefficients for V (B)
    # V[j, k, r]
    V_s = [
        [[1, 1, 0, -1, 0, 1, 0],  # j=0, k=0 (b00)
         [0, 0, 1, 0, 0, 1, 0]],  # j=0, k=1 (b01)
        [[1, 0, -1, 1, 1, 0, 1],  # j=1, k=0 (b10)
         [1, 0, 0, 0, 1, 0, 1]]   # j=1, k=1 (b11)
    ]
    
    # Coefficients for W (C)
    # W[k, i, r]
    W_s = [
        [[1, 0, 0, 1, -1, 0, 1],  # k=0, i=0 (c00)
         [0, 0, 1, 0, 1, 0, 0]],  # k=0, i=1 (c01)
        [[0, 1, 0, 1, 0, 0, 0],   # k=1, i=0 (c10)
         [1, -1, 1, 0, 0, 1, 0]]  # k=1, i=1 (c11)
    ]
    
    # Define the 4 blocks
    # Block 1: A_L B_TL -> C_L
    # A_L: i in {0,1}, j in {0,1}
    # B_TL: j in {0,1}, k in {0,1}
    # C_L: k in {0,1}, i in {0,1}
    
    # Block 2: A_R B_BL -> C_L
    # A_R: i in {0,1}, j in {2,3}
    # B_BL: j in {2,3}, k in {0,1}
    # C_L: k in {0,1}, i in {0,1}
    
    # Block 3: A_L B_TR -> C_R
    # A_L: i in {0,1}, j in {0,1}
    # B_TR: j in {0,1}, k in {2,3}
    # C_R: k in {2,3}, i in {0,1}
    
    # Block 4: A_R B_BR -> C_R
    # A_R: i in {0,1}, j in {2,3}
    # B_BR: j in {2,3}, k in {2,3}
    # C_R: k in {2,3}, i in {0,1}
    
    blocks = [
        # (j_offset_A, j_offset_B, k_offset_C, col_offset)
        (0, 0, 0, 0),    # Block 1
        (2, 2, 0, 7),    # Block 2
        (0, 0, 2, 14),   # Block 3
        (2, 2, 2, 21)    # Block 4
    ]
    
    for j_off_A, j_off_B, k_off_C, col_off in blocks:
        for i in range(2):
            for j_local in range(2):
                j_global = j_off_A + j_local
                u_row = i * m + j_global
                for r_local in range(7):
                    U[u_row][col_off + r_local] = U_s[i][j_local][r_local]
        
        for j_local in range(2):
            j_global = j_off_B + j_local
            for k_local in range(2):
                k_global = j_global # No, k is column index of B, which is row index of C?
                # V is indexed by j (row of B) and k (col of B).
                # B is 4x4. Rows 0-3, Cols 0-3.
                # j_global is row of B.
                # k_global is col of B.
                # In Block 1, B_TL is rows 0-1, cols 0-1.
                # So j_global in {0,1}, k_global in {0,1}.
                # In Block 2, B_BL is rows 2-3, cols 0-1.
                # So j_global in {2,3}, k_global in {0,1}.
                # j_off_B is offset for rows of B.
                # k_offset for cols of B is not separate, it's determined by block.
                # Wait, my block definition was (j_off_A, j_off_B, k_off_C, col_off).
                # I need to be careful with k.
                # For Block 1: j in {0,1}, k in {0,1}.
                # For Block 2: j in {2,3}, k in {0,1}.
                # For Block 3: j in {0,1}, k in {2,3}.
                # For Block 4: j in {2,3}, k in {2,3}.
                
                # So I need j_off_B and k_off_B.
                # Let's redefine blocks with (j_off_A, j_off_B, k_off_B, k_off_C, col_off)
                
                pass
        # Redoing the loop structure to be clearer
        
    # Let's restart the filling logic with clearer offsets
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Blocks: (j_off_A, j_off_B, k_off_B, k_off_C, col_off)
    # j_off_A: offset for columns of A (which are rows of B)
    # j_off_B: offset for rows of B
    # k_off_B: offset for columns of B
    # k_off_C: offset for columns of C (which are rows of W)
    
    blocks = [
        (0, 0, 0, 0, 0),    # B1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        (2, 2, 0, 0, 7),    # B2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        (0, 0, 2, 2, 14),   # B3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        (2, 2, 2, 2, 21)    # B4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    ]
    
    for j_off_A, j_off_B, k_off_B, k_off_C, col_off in blocks:
        # Fill U
        for i in range(2):
            for j_l in range(2):
                j_g = j_off_A + j_l
                u_row = i * m + j_g
                for r_l in range(7):
                    U[u_row][col_off + r_l] = U_s[i][j_l][r_l]
        
        # Fill V
        for j_l in range(2):
            j_g = j_off_B + j_l
            for k_l in range(2):
                k_g = k_off_B + k_l
                v_row = j_g * p + k_g
                for r_l in range(7):
                    V[v_row][col_off + r_l] = V_s[j_l][k_l][r_l]
        
        # Fill W
        for k_l in range(2):
            k_g = k_off_C + k_l
            for i in range(2):
                w_row = k_g * n + i
                for r_l in range(7):
                    W[w_row][col_off + r_l] += W_s[k_l][i][r_l]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
