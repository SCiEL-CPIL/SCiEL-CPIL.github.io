"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Base Strassen decomposition for 2x2 x 2x2 -> 2x2
    # U_base, V_base are 4x7, W_base is 4x7
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U_base = [
        [1, 0, 1, 0, 1, 0, 1], # a
        [0, 0, 0, 1, 1, 0, -1], # b
        [0, 1, 0, 0, 0, 1, 0], # c
        [1, 1, 0, 0, 0, 1, 0]  # d
    ]
    
    V_base = [
        [1, 0, 0, 1, 1, 0, 0], # e
        [0, 0, 1, -1, 0, 0, 1], # f
        [0, 0, 0, 0, 0, 1, 0], # g
        [1, 1, -1, 0, 0, 1, 1]  # h
    ]
    
    W_base = [
        [1, 0, 0, 1, -1, 1, 0], # C00
        [0, 1, 0, 1, 0, 0, 0], # C01
        [0, 0, 1, 0, 1, 0, 0], # C10
        [1, 0, -1, 1, 0, 0, 1]  # C11
    ]

    # Define the 7 Strassen terms for 4x4 x 4x4
    # Each term k defines:
    # left_blocks: dict mapping local_pos (0..3) to list of (global_row, global_col, coeff)
    # right_blocks: dict mapping local_pos (0..3) to list of (global_row, global_col, coeff)
    # out_coeffs: dict mapping out_block (0..3) to coeff
    
    # Local pos: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Global A blocks:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    def get_block_indices(block):
        # Returns list of 4 global indices (row*4+col) for the 2x2 block
        if block == 0: # A00
            return [0*4+0, 0*4+1, 1*4+0, 1*4+1]
        elif block == 1: # A01
            return [0*4+2, 0*4+3, 1*4+2, 1*4+3]
        elif block == 2: # A10
            return [2*4+0, 2*4+1, 3*4+0, 3*4+1]
        elif block == 3: # A11
            return [2*4+2, 2*4+3, 3*4+2, 3*4+3]

    terms = []
    
    # Term 0: (A00 + A11) * (B00 + B11)
    # Left: A00 + A11. Local pos r corresponds to block A00[r] + A11[r]
    # Right: B00 + B11. Local pos r corresponds to block B00[r] + B11[r]
    # Out: C00 += 1, C11 += 1
    terms.append({
        'left': {r: [(get_block_indices(0)[r], 1), (get_block_indices(3)[r], 1)] for r in range(4)},
        'right': {r: [(get_block_indices(0)[r], 1), (get_block_indices(3)[r], 1)] for r in range(4)},
        'out': {0: 1, 3: 1}
    })
    
    # Term 1: (A10 + A11) * B11
    # Left: A10 + A11
    # Right: B11
    # Out: C01 += 1
    terms.append({
        'left': {r: [(get_block_indices(2)[r], 1), (get_block_indices(3)[r], 1)] for r in range(4)},
        'right': {r: [(get_block_indices(3)[r], 1)] for r in range(4)},
        'out': {1: 1}
    })
    
    # Term 2: A00 * (B01 - B11)
    # Left: A00
    # Right: B01 - B11
    # Out: C10 += 1
    terms.append({
        'left': {r: [(get_block_indices(0)[r], 1)] for r in range(4)},
        'right': {r: [(get_block_indices(1)[r], 1), (get_block_indices(3)[r], -1)] for r in range(4)},
        'out': {2: 1}
    })
    
    # Term 3: A01 * (B00 - B10)
    # Left: A01
    # Right: B00 - B10
    # Out: C00 += 1, C01 += 1, C11 += 1
    terms.append({
        'left': {r: [(get_block_indices(1)[r], 1)] for r in range(4)},
        'right': {r: [(get_block_indices(0)[r], 1), (get_block_indices(2)[r], -1)] for r in range(4)},
        'out': {0: 1, 1: 1, 3: 1}
    })
    
    # Term 4: (A00 + A01) * B10
    # Left: A00 + A01
    # Right: B10
    # Out: C00 -= 1, C10 += 1
    terms.append({
        'left': {r: [(get_block_indices(0)[r], 1), (get_block_indices(1)[r], 1)] for r in range(4)},
        'right': {r: [(get_block_indices(2)[r], 1)] for r in range(4)},
        'out': {0: -1, 2: 1}
    })
    
    # Term 5: (A10 - A00) * (B00 + B01)
    # Left: A10 - A00
    # Right: B00 + B01
    # Out: C00 += 1
    terms.append({
        'left': {r: [(get_block_indices(2)[r], 1), (get_block_indices(0)[r], -1)] for r in range(4)},
        'right': {r: [(get_block_indices(0)[r], 1), (get_block_indices(1)[r], 1)] for r in range(4)},
        'out': {0: 1}
    })
    
    # Term 6: (A11 - A01) * (B10 + B11)
    # Left: A11 - A01
    # Right: B10 + B11
    # Out: C11 += 1
    terms.append({
        'left': {r: [(get_block_indices(3)[r], 1), (get_block_indices(1)[r], -1)] for r in range(4)},
        'right': {r: [(get_block_indices(2)[r], 1), (get_block_indices(3)[r], 1)] for r in range(4)},
        'out': {3: 1}
    })

    # Construct 4x4x4 decomposition
    rank_444 = 49
    U_444 = [[0] * rank_444 for _ in range(16)]
    V_444 = [[0] * rank_444 for _ in range(16)]
    W_444 = [[0] * rank_444 for _ in range(16)]
    
    col = 0
    for k in range(7):
        term = terms[k]
        for s in range(7):
            # Update U
            for local_r in range(4):
                val_base = U_base[local_r][s]
                if val_base != 0:
                    for global_idx, coeff in term['left'][local_r]:
                        U_444[global_idx][col] += val_base * coeff
                        
            # Update V
            for local_r in range(4):
                val_base = V_base[local_r][s]
                if val_base != 0:
                    for global_idx, coeff in term['right'][local_r]:
                        V_444[global_idx][col] += val_base * coeff
                        
            # Update W
            for local_r in range(4):
                val_base = W_base[local_r][s]
                if val_base != 0:
                    # local_r 0->C00, 1->C01, 2->C10, 3->C11
                    # We need to map local_r to the block index in C
                    # And then to global indices
                    block_idx = local_r
                    coeff_out = term['out'].get(block_idx, 0)
                    if coeff_out != 0:
                        global_indices = get_block_indices(block_idx)
                        # The local_r in W_base corresponds to the output block's local position?
                        # No, W_base rows are 0..3 corresponding to C00, C01, C10, C11.
                        # So local_r is the block index.
                        # But we need to distribute to the 4 global rows of that block.
                        # Wait, W_base[local_r][s] is the coefficient for the entire block?
                        # No, W_base is for 2x2 output.
                        # In the 4x4 case, the output of term k is a 2x2 matrix T_k.
                        # T_k is added to C.
                        # The "output" of the 2x2 Strassen is a 2x2 matrix.
                        # The rows of W_base correspond to the 4 entries of this 2x2 matrix.
                        # So local_r 0 is T_k[0,0], local_r 1 is T_k[0,1], etc.
                        # T_k[0,0] is added to C00[0,0].
                        # T_k[0,1] is added to C00[0,1].
                        # So we need to map local_r to the specific global entry.
                        
                        # local_r 0 -> C_block[0,0]
                        # local_r 1 -> C_block[0,1]
                        # local_r 2 -> C_block[1,0]
                        # local_r 3 -> C_block[1,1]
                        
                        # Get global indices for the block
                        g_indices = get_block_indices(block_idx)
                        global_idx = g_indices[local_r]
                        
                        W_444[global_idx][col] += val_base * coeff_out
                        
            col += 1

    # Extend to 4x4x8
    # B is 4x8. Split into B_L (cols 0-3) and B_R (cols 4-7).
    # C is 4x8. Split into C_L (cols 0-3) and C_R (cols 4-7).
    
    rank = 98
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    # Left half (cols 0-49)
    for i in range(16):
        for c in range(49):
            U[i][c] = U_444[i][c]
            V[i][c] = V_444[i][c] # B_L is rows 0-15 of V
            W[i][c] = W_444[i][c] # C_L is rows 0-15 of W
            
    # Right half (cols 49-98)
    for i in range(16):
        for c in range(49):
            U[i][c + 49] = U_444[i][c]
            # B_R is rows 16-31 of V. Map B_444 row i to V row i+16
            V[i + 16][c + 49] = V_444[i][c]
            # C_R is rows 16-31 of W. Map C_444 row i to W row i+16
            W[i + 16][c + 49] = W_444[i][c]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
