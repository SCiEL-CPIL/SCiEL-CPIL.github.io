"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # We will construct a rank-98 decomposition based on recursive Strassen for 4x4.
    # Rank 49 for 4x4x4, duplicated for the two 4x4 blocks of the 4x4x8 problem.
    
    # 1. Define the 2x2 Strassen decomposition
    # Indices: 0:00, 1:01, 2:10, 3:11
    # U_2x2: 4x7
    # V_2x2: 4x7
    # W_2x2: 4x7
    
    U_2 = np.zeros((4, 7), dtype=int)
    V_2 = np.zeros((4, 7), dtype=int)
    W_2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1 # C00, C11 get +M1
    
    # M2 = (A10+A11)B00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1 # C10 gets +M2, C11 gets -M2
    
    # M3 = A00(B01-B11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1 # C01 gets +M3, C11 gets +M3
    
    # M4 = A11(B10-B00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1 # C00 gets +M4, C10 gets +M4
    
    # M5 = (A00+A01)B11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1 # C00 gets -M5, C01 gets +M5
    
    # M6 = (A10-A00)(B00+B01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1 # C11 gets +M6
    
    # M7 = (A01-A11)(B10+B11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1; W_2[3, 6] = -1 # C00 gets +M7, C11 gets -M7
    
    # 2. Construct 4x4 decomposition using 7 blocks of 2x2
    # A is 4x4, split into 2x2 blocks A00, A01, A10, A11
    # B is 4x4, split into 2x2 blocks B00, B01, B10, B11
    # C is 4x4, split into 2x2 blocks C00, C01, C10, C11
    
    # The 7 intermediate products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mk is a 2x2 matrix product.
    # We expand each Mk using the 2x2 Strassen decomposition (rank 7).
    # Total rank = 7 * 7 = 49.
    
    # We need to map the 4x4 indices to the block structure.
    # A index (i, j) where i, j in 0..3.
    # Block row = i // 2, Block col = j // 2.
    # Local row = i % 2, Local col = j % 2.
    
    # Let's build U_4, V_4, W_4 of size 16x49.
    U_4 = np.zeros((16, 49), dtype=int)
    V_4 = np.zeros((16, 49), dtype=int)
    W_4 = np.zeros((16, 49), dtype=int)
    
    # Define the 7 block-level multiplications
    # Each is a pair of (A_block, B_block) with coefficients
    # A_block is a linear combination of A's 2x2 blocks.
    # B_block is a linear combination of B's 2x2 blocks.
    
    # A_blocks: 00, 01, 10, 11
    # B_blocks: 00, 01, 10, 11
    
    block_defs = [
        # (A_coeffs, B_coeffs)
        # A_coeffs: dict {block_idx: coeff}
        # B_coeffs: dict {block_idx: coeff}
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {1: 1, 3: -1}),      # M3
        ({3: 1}, {2: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
    ]
    
    col_offset = 0
    for k in range(7):
        A_coeffs, B_coeffs = block_defs[k]
        
        # For each of the 7 rank-1 terms in the 2x2 decomposition:
        for t in range(7):
            col = col_offset + t
            
            # Get the local 2x2 coefficients for U, V, W
            # U_2[local_idx, t]
            # V_2[local_idx, t]
            # W_2[local_idx, t]
            
            # Fill U_4
            # A index = block_idx * 2 + local_idx? No.
            # Block idx 0 is top-left (0,0) -> A indices 0,1,4,5
            # Block idx 1 is top-right (0,1) -> A indices 2,3,6,7
            # Block idx 2 is bottom-left (1,0) -> A indices 8,9,12,13
            # Block idx 3 is bottom-right (1,1) -> A indices 10,11,14,15
            
            # Local idx 0: (0,0) -> offset 0
            # Local idx 1: (0,1) -> offset 1
            # Local idx 2: (1,0) -> offset 4
            # Local idx 3: (1,1) -> offset 5
            
            # Wait, A is 4x4.
            # A[i,j] is at index i*4 + j.
            # Block (br, bc) covers rows 2br..2br+1, cols 2bc..2bc+1.
            # Local (lr, lc) corresponds to row 2br+lr, col 2bc+lc.
            # Global index = (2br+lr)*4 + (2bc+lc).
            
            # For U_4 (A part):
            for br in range(2):
                for bc in range(2):
                    block_idx = br * 2 + bc
                    coeff_A = A_coeffs.get(block_idx, 0)
                    if coeff_A != 0:
                        for lr in range(2):
                            for lc in range(2):
                                local_idx = lr * 2 + lc
                                u_val = U_2[local_idx, t]
                                if u_val != 0:
                                    global_idx = (2*br + lr) * 4 + (2*bc + lc)
                                    U_4[global_idx, col] += coeff_A * u_val
            
            # For V_4 (B part):
            # Same indexing for B
            for br in range(2):
                for bc in range(2):
                    block_idx = br * 2 + bc
                    coeff_B = B_coeffs.get(block_idx, 0)
                    if coeff_B != 0:
                        for lr in range(2):
                            for lc in range(2):
                                local_idx = lr * 2 + lc
                                v_val = V_2[local_idx, t]
                                if v_val != 0:
                                    global_idx = (2*br + lr) * 4 + (2*bc + lc)
                                    V_4[global_idx, col] += coeff_B * v_val
            
            # For W_4 (C part):
            # C is 4x4. Same indexing.
            for lr in range(2):
                for lc in range(2):
                    local_idx = lr * 2 + lc
                    w_val = W_2[local_idx, t]
                    if w_val != 0:
                        # C index mapping:
                        # C[i,j] where i,j in 0..3.
                        # The output of the 2x2 Strassen is C_local[local_idx].
                        # This C_local corresponds to the entire 4x4 C?
                        # No, the block-level M_k produces a 2x2 block of C.
                        # Which block?
                        # C = A B.
                        # C_block(br, bc) = sum_b A_block(br, b) B_block(b, bc).
                        # The 7 M_k are not directly C blocks.
                        # They are intermediate products.
                        # We need to express C blocks in terms of M_k.
                        
                        # C00 = M1 + M4 - M5 + M7
                        # C01 = M3 + M5
                        # C10 = M2 + M4
                        # C11 = M1 - M2 + M3 + M6
                        
                        # So for each M_k, we add its contribution to the specific C blocks.
                        pass # Handled below

    # The above loop calculated U_4 and V_4 correctly for the intermediate products M_k.
    # Now we need to map M_k to C.
    # Instead of building W_4 in the loop, we can build it afterwards or modify the loop.
    # Let's restart the W_4 construction.
    
    W_4 = np.zeros((16, 49), dtype=int)
    
    c_block_map = {
        0: [1, 0, 0, 1, -1, 0, 1],   # C00: M1, M4, -M5, M7
        1: [0, 0, 1, 0, 1, 0, 0],    # C01: M3, M5
        2: [0, 1, 0, 1, 0, 0, 0],    # C10: M2, M4
        3: [1, -1, 1, 0, 0, 1, -1]   # C11: M1, -M2, M3, M6, -M7
    }
    
    col_offset = 0
    for k in range(7):
        for t in range(7):
            col = col_offset + t
            # W_4 for this column is determined by which C blocks M_k contributes to.
            for cb in range(4):
                coeff = c_block_map[cb][k]
                if coeff != 0:
                    # M_k contributes coeff * M_k to C_block(cb)
                    # M_k is a 2x2 matrix.
                    # We use the 2x2 Strassen W_2 to map M_k's local indices to C_block's local indices.
                    # C_block(cb) is at global position (cb_row, cb_col).
                    # cb 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1).
                    cb_row = cb // 2
                    cb_col = cb % 2
                    
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            w_val = W_2[local_idx, t]
                            if w_val != 0:
                                global_idx = (2*cb_row + lr) * 4 + (2*cb_col + lc)
                                W_4[global_idx, col] += coeff * w_val
            col_offset += 1

    # Now we have U_4, V_4, W_4 for 4x4x4.
    # We need to extend to 4x4x8.
    # B is 4x8. C is 4x8.
    # B = [B1, B2], C = [C1, C2].
    # C1 = A B1, C2 = A B2.
    # We use the same U_4 for both.
    # V is 32x49. V[0:16] = V_4, V[16:32] = V_4.
    # W is 32x49. W[0:16] = W_4, W[16:32] = W_4.
    
    U = U_4.tolist()
    V = np.vstack([V_4, V_4]).tolist()
    W = np.vstack([W_4, W_4]).tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
