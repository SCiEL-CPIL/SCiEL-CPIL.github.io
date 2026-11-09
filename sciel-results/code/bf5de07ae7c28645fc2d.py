"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # Safety margin
    
    n, m, p = DIMENSIONS # 4, 4, 8
    
    # Strassen 2x2x2 decomposition
    # U: 4x7, V: 4x7, W: 4x7
    # Local indices:
    # U: 0:A00, 1:A01, 2:A10, 3:A11
    # V: 0:B00, 1:B01, 2:B10, 3:B11
    # W: 0:C00, 1:C01, 2:C10, 3:C11
    
    U_2x2 = np.zeros((4, 7))
    V_2x2 = np.zeros((4, 7))
    W_2x2 = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11)
    U_2x2[0, 0] = 1; U_2x2[3, 0] = 1
    V_2x2[0, 0] = 1; V_2x2[3, 0] = 1
    W_2x2[0, 0] = 1; W_2x2[3, 0] = 1
    
    # Term 2: (A10+A11)B00
    U_2x2[2, 1] = 1; U_2x2[3, 1] = 1
    V_2x2[0, 1] = 1
    W_2x2[2, 1] = 1; W_2x2[3, 1] = -1
    
    # Term 3: A00(B01-B11)
    U_2x2[0, 2] = 1
    V_2x2[1, 2] = 1; V_2x2[3, 2] = -1
    W_2x2[1, 2] = 1; W_2x2[3, 2] = 1
    
    # Term 4: A11(B10-B00)
    U_2x2[3, 3] = 1
    V_2x2[2, 3] = 1; V_2x2[0, 3] = -1
    W_2x2[0, 3] = 1; W_2x2[2, 3] = 1
    
    # Term 5: (A00+A01)B11
    U_2x2[0, 4] = 1; U_2x2[1, 4] = 1
    V_2x2[3, 4] = 1
    W_2x2[0, 4] = -1; W_2x2[1, 4] = 1
    
    # Term 6: (A10-A00)(B00+B01)
    U_2x2[2, 5] = 1; U_2x2[0, 5] = -1
    V_2x2[0, 5] = 1; V_2x2[1, 5] = 1
    W_2x2[3, 5] = 1
    
    # Term 7: (A01-A11)(B10+B11)
    U_2x2[1, 6] = 1; U_2x2[3, 6] = -1
    V_2x2[2, 6] = 1; V_2x2[3, 6] = 1
    W_2x2[0, 6] = 1
    
    rank = 98
    U = np.zeros((n*m, rank)) # 16x98
    V = np.zeros((m*p, rank)) # 32x98
    W = np.zeros((p*n, rank)) # 32x98
    
    # Mapping for 4x4x4 blocks
    # A blocks (U indices):
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    A_block_map = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    
    # B blocks (V indices) for 4x4:
    # B00: 0,1,4,5
    # B01: 2,3,6,7
    # B10: 8,9,12,13
    # B11: 10,11,14,15
    B_block_map = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    
    # C blocks (W indices) for 4x4:
    # C00: 0,1,4,5
    # C01: 8,9,12,13
    # C10: 2,3,6,7
    # C11: 10,11,14,15
    C_block_map = {
        0: [0, 1, 4, 5],
        1: [8, 9, 12, 13],
        2: [2, 3, 6, 7],
        3: [10, 11, 14, 15]
    }
    
    # Strassen 4x4x4 structure
    # 7 intermediate matrices M1..M7
    # Each Mk is 2x2.
    # Mk is formed by 2x2x2 Strassen on blocks.
    
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # For each Mk, we apply the 2x2x2 Strassen decomposition.
    # The inputs to the 2x2x2 are the blocks of A and B.
    # The output of the 2x2x2 is the 2x2 matrix Mk.
    
    # We need to map the local 2x2 indices to global 4x4 indices.
    
    # For Block 0 (cols 0-3 of B)
    # V indices are 0..15
    # W indices are 0..15
    
    # For Block 1 (cols 4-7 of B)
    # V indices are 16..31 (offset 16)
    # W indices are 16..31 (offset 16)
    
    # Let's build Block 0
    col_offset = 0
    for k in range(7):
        # Determine A and B blocks for Mk
        # k=0: M1 -> A00,A11 and B00,B11
        # k=1: M2 -> A10,A11 and B00
        # k=2: M3 -> A00 and B01,B11
        # k=3: M4 -> A11 and B10,B00
        # k=4: M5 -> A00,A01 and B11
        # k=5: M6 -> A10,A00 and B00,B01
        # k=6: M7 -> A01,A11 and B10,B11
        
        # Define which A blocks are involved and their signs
        # And which B blocks are involved and their signs
        # This is effectively the U and V of the 2x2x2 Strassen, but applied to blocks.
        
        # We can reuse the U_2x2 and V_2x2 logic.
        # U_2x2[i, t] gives coeff of A_i in term t.
        # V_2x2[j, t] gives coeff of B_j in term t.
        
        for t in range(7):
            # This term t of Mk
            global_col = col_offset + k*7 + t
            
            # Fill U
            for i_local in range(4):
                coeff = U_2x2[i_local, t]
                if coeff != 0:
                    # i_local maps to A block i_local
                    # A block i_local has 4 global indices
                    for idx in A_block_map[i_local]:
                        U[idx, global_col] = coeff
            
            # Fill V
            for j_local in range(4):
                coeff = V_2x2[j_local, t]
                if coeff != 0:
                    # j_local maps to B block j_local
                    for idx in B_block_map[j_local]:
                        V[idx, global_col] = coeff
            
            # Fill W
            # W_2x2[l, t] gives coeff for local output l in term t.
            # Local output l maps to a position in the 2x2 matrix Mk.
            # Mk contributes to specific C blocks.
            
            # Mapping of Mk to C blocks:
            # M1 (k=0): C00 (+1), C11 (+1)
            # M2 (k=1): C10 (+1), C11 (-1)
            # M3 (k=2): C01 (+1), C11 (+1)
            # M4 (k=3): C00 (+1), C10 (+1)
            # M5 (k=4): C00 (-1), C01 (+1)
            # M6 (k=5): C11 (+1)
            # M7 (k=6): C00 (+1)
            
            # For each local output position l (0..3) in the 2x2 block Mk:
            # l=0: (0,0)
            # l=1: (0,1)
            # l=2: (1,0)
            # l=3: (1,1)
            
            # We need to add W_2x2[l, t] * coeff_to_C_block to the corresponding C index.
            
            # First, determine which C blocks Mk contributes to and with what coefficient.
            # This is a property of k.
            # Let's define a list of (C_block_id, coeff) for each k.
            
            if k == 0: # M1
                targets = [(0, 1), (3, 1)] # C00, C11
            elif k == 1: # M2
                targets = [(2, 1), (3, -1)] # C10, C11
            elif k == 2: # M3
                targets = [(1, 1), (3, 1)] # C01, C11
            elif k == 3: # M4
                targets = [(0, 1), (2, 1)] # C00, C10
            elif k == 4: # M5
                targets = [(0, -1), (1, 1)] # C00, C01
            elif k == 5: # M6
                targets = [(3, 1)] # C11
            elif k == 6: # M7
                targets = [(0, 1)] # C00
            
            for l in range(4):
                w_coeff = W_2x2[l, t]
                if w_coeff == 0: continue
                
                # l maps to local (r, c_loc)
                r = l // 2
                c_loc = l % 2
                
                for c_block, c_coeff in targets:
                    total_coeff = w_coeff * c_coeff
                    if total_coeff == 0: continue
                    
                    # c_block is 0..3
                    # r, c_loc are 0..1
                    # Global C index for block c_block, pos (r, c_loc)
                    # c_block 0: C00. Rows 0,1. Cols 0,1.
                    # c_block 1: C01. Rows 0,1. Cols 2,3.
                    # c_block 2: C10. Rows 2,3. Cols 0,1.
                    # c_block 3: C11. Rows 2,3. Cols 2,3.
                    
                    # Block row offset: 0 for C00,C01. 2 for C10,C11.
                    # Block col offset: 0 for C00,C10. 2 for C01,C11.
                    
                    if c_block == 0: # C00
                        row_off = 0; col_off = 0
                    elif c_block == 1: # C01
                        row_off = 0; col_off = 2
                    elif c_block == 2: # C10
                        row_off = 2; col_off = 0
                    elif c_block == 3: # C11
                        row_off = 2; col_off = 2
                    
                    global_row = row_off + r
                    global_col_c = col_off + c_loc
                    
                    # W index: 4 * global_col_c + global_row
                    w_idx = 4 * global_col_c + global_row
                    
                    W[w_idx, global_col] = total_coeff
    
    # Build Block 1 (cols 4-7 of B)
    # V indices offset by 16
    # W indices offset by 16
    col_offset = 49
    for k in range(7):
        for t in range(7):
            global_col = col_offset + k*7 + t
            
            # Fill U (same as before)
            for i_local in range(4):
                coeff = U_2x2[i_local, t]
                if coeff != 0:
                    for idx in A_block_map[i_local]:
                        U[idx, global_col] = coeff
            
            # Fill V (offset by 16)
            for j_local in range(4):
                coeff = V_2x2[j_local, t]
                if coeff != 0:
                    for idx in B_block_map[j_local]:
                        V[idx + 16, global_col] = coeff
            
            # Fill W (offset by 16)
            if k == 0: # M1
                targets = [(0, 1), (3, 1)]
            elif k == 1: # M2
                targets = [(2, 1), (3, -1)]
            elif k == 2: # M3
                targets = [(1, 1), (3, 1)]
            elif k == 3: # M4
                targets = [(0, 1), (2, 1)]
            elif k == 4: # M5
                targets = [(0, -1), (1, 1)]
            elif k == 5: # M6
                targets = [(3, 1)]
            elif k == 6: # M7
                targets = [(0, 1)]
            
            for l in range(4):
                w_coeff = W_2x2[l, t]
                if w_coeff == 0: continue
                
                r = l // 2
                c_loc = l % 2
                
                for c_block, c_coeff in targets:
                    total_coeff = w_coeff * c_coeff
                    if total_coeff == 0: continue
                    
                    if c_block == 0:
                        row_off = 0; col_off = 0
                    elif c_block == 1:
                        row_off = 0; col_off = 2
                    elif c_block == 2:
                        row_off = 2; col_off = 0
                    elif c_block == 3:
                        row_off = 2; col_off = 2
                    
                    global_row = row_off + r
                    global_col_c = col_off + c_loc
                    
                    w_idx = 4 * global_col_c + global_row + 16
                    
                    W[w_idx, global_col] = total_coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
