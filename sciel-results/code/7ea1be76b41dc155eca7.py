"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # U shape: (n*m, rank) = (4, 49)
    # V shape: (m*p, rank) = (28, 49)
    # W shape: (p*n, rank) = (28, 49)
    
    # Strassen coefficients for 2x2 multiplication
    # U indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # V local indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # W local indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (corresponding to C_00, C_01, C_10, C_11)
    
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, -1, 1, 0], # M6
        [0, 1, -1, 0]  # M7
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    
    strassen_W = [
        [1, 0, 0, 0], # C00
        [0, 1, 0, 0], # C01
        [0, 0, 1, 0], # C10
        [0, 0, 0, 1]  # C11
    ]
    
    # Reconstruction coefficients: W[output_index][strassen_step]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    recon = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    num_blocks = p // 2 # 7 blocks
    rank = 7 * num_blocks # 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(num_blocks):
        # Block b uses columns 2b and 2b+1 of B
        # and rows 2b and 2b+1 of C
        
        for s in range(7):
            # U part is the same for all blocks
            for i in range(4):
                U[i][col] = strassen_U[s][i]
            
            # V part
            # B indices for this block:
            # B[0, 2b] -> index 0*14 + 2b
            # B[0, 2b+1] -> index 0*14 + 2b + 1
            # B[1, 2b] -> index 1*14 + 2b
            # B[1, 2b+1] -> index 1*14 + 2b + 1
            
            v_local = strassen_V[s]
            # v_local[0] is coeff for B00 (row 0, col 0 of block)
            # v_local[1] is coeff for B01 (row 0, col 1 of block)
            # v_local[2] is coeff for B10 (row 1, col 0 of block)
            # v_local[3] is coeff for B11 (row 1, col 1 of block)
            
            V[0 * p + 2 * b][col] = v_local[0]
            V[0 * p + 2 * b + 1][col] = v_local[1]
            V[1 * p + 2 * b][col] = v_local[2]
            V[1 * p + 2 * b + 1][col] = v_local[3]
            
            # W part
            # C indices for this block:
            # C[2b, 0] -> index 2b * n + 0
            # C[2b, 1] -> index 2b * n + 1
            # C[2b+1, 0] -> index (2b+1) * n + 0
            # C[2b+1, 1] -> index (2b+1) * n + 1
            
            # recon[0] is coeffs for C00
            # recon[1] is coeffs for C01
            # recon[2] is coeffs for C10
            # recon[3] is coeffs for C11
            
            w_local = [recon[0][s], recon[1][s], recon[2][s], recon[3][s]]
            
            W[2 * b * n + 0][col] = w_local[0]
            W[2 * b * n + 1][col] = w_local[1]
            W[(2 * b + 1) * n + 0][col] = w_local[2]
            W[(2 * b + 1) * n + 1][col] = w_local[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
