"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # We construct a hybrid decomposition:
    # 1. Use Strassen's algorithm for the 4x4 core (rank 49).
    # 2. Use schoolbook for the remaining terms involving index 4 (rank 61).
    # Total rank = 49 + 61 = 110.
    
    # --- Part 1: Strassen decomposition for 4x4 core ---
    # The 4x4 matrix multiplication tensor can be decomposed into 49 rank-1 terms
    # by applying Strassen's 2x2 algorithm to the 2x2 blocks of the 4x4 matrices.
    
    # Local 2x2 Strassen coefficients (U, V, W) for the 7 multiplications.
    # Indices: U/V are 4-dim (a,b,c,d / e,f,g,h), W is 4-dim (C00,C01,C10,C11)
    strassen_2x2 = [
        # U, V, W
        ([1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]), # m1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # m2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # m3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # m4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # m5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # m6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # m7
    ]
    
    # Block Strassen coefficients for 4x4 (mapping 4x4 blocks to 2x2 block ops)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We define the linear combinations for X and Y for each of the 7 block multiplications.
    # X and Y are 2x2 matrices.
    # We represent the linear combination of 4x4 entries (16 entries) for each 2x2 entry.
    # Actually, it's easier to map local 2x2 indices to global 4x4 indices.
    
    # For each of the 7 block multiplications, we define:
    # x_map: dict mapping (u,v) to list of (global_i, global_j, coeff)
    # y_map: dict mapping (u,v) to list of (global_i, global_j, coeff)
    # c_block: which block of C (00, 01, 10, 11) this contributes to, with sign
    
    # Block indices: 00 (0,0), 01 (0,1), 10 (1,0), 11 (1,1)
    # Global indices: i = block_i * 2 + local_i, j = block_j * 2 + local_j
    
    block_defs = [
        # M1
        {
            'x': { (0,0): [(0,0,1), (2,2,1)], (0,1): [(0,1,1), (2,3,1)], (1,0): [(1,0,1), (3,2,1)], (1,1): [(1,1,1), (3,3,1)] },
            'y': { (0,0): [(0,0,1), (2,2,1)], (0,1): [(0,1,1), (2,3,1)], (1,0): [(1,0,1), (2,3,1)], (1,1): [(1,1,1), (3,3,1)] },
            'c_blocks': { 0: 1, 1: 0, 2: 0, 3: 1 } # C00, C01, C10, C11 coeffs
        },
        # M2
        {
            'x': { (0,0): [(2,0,1), (0,0,1)], (0,1): [(2,1,1), (0,1,1)], (1,0): [(3,0,1), (1,0,1)], (1,1): [(3,1,1), (1,1,1)] },
            'y': { (0,0): [(0,0,1)], (0,1): [(0,1,1)], (1,0): [(1,0,1)], (1,1): [(1,1,1)] },
            'c_blocks': { 0: 0, 1: 0, 2: 1, 3: -1 }
        },
        # M3
        {
            'x': { (0,0): [(0,0,1)], (0,1): [(0,1,1)], (1,0): [(1,0,1)], (1,1): [(1,1,1)] },
            'y': { (0,0): [(0,2,-1), (2,2,-1)], (0,1): [(0,3,-1), (2,3,-1)], (1,0): [(1,2,-1), (3,2,-1)], (1,1): [(1,3,-1), (3,3,-1)] },
            'c_blocks': { 0: 0, 1: 1, 2: 0, 3: 1 }
        },
        # M4
        {
            'x': { (0,0): [(2,2,1)], (0,1): [(2,3,1)], (1,0): [(3,2,1)], (1,1): [(3,3,1)] },
            'y': { (0,0): [(0,0,-1)], (0,1): [(0,1,-1)], (1,0): [(1,0,1)], (1,1): [(1,1,1)] },
            'c_blocks': { 0: 1, 1: 0, 2: 1, 3: 0 }
        },
        # M5
        {
            'x': { (0,0): [(0,0,1), (0,2,1)], (0,1): [(0,1,1), (0,3,1)], (1,0): [(1,0,1), (1,2,1)], (1,1): [(1,1,1), (1,3,1)] },
            'y': { (0,0): [(2,2,1)], (0,1): [(2,3,1)], (1,0): [(3,2,1)], (1,1): [(3,3,1)] },
            'c_blocks': { 0: -1, 1: 1, 2: 0, 3: 0 }
        },
        # M6
        {
            'x': { (0,0): [(2,0,1), (0,0,-1)], (0,1): [(2,1,1), (0,1,-1)], (1,0): [(3,0,1), (1,0,-1)], (1,1): [(3,1,1), (1,1,-1)] },
            'y': { (0,0): [(0,0,1)], (0,1): [(0,1,1)], (1,0): [(1,0,1)], (1,1): [(1,1,1)] },
            'c_blocks': { 0: 0, 1: 0, 2: 0, 3: 1 }
        },
        # M7
        {
            'x': { (0,0): [(0,2,1), (2,2,-1)], (0,1): [(0,3,1), (2,3,-1)], (1,0): [(1,2,1), (3,2,-1)], (1,1): [(1,3,1), (3,3,-1)] },
            'y': { (0,0): [(2,0,1)], (0,1): [(2,1,1)], (1,0): [(3,0,1)], (1,1): [(3,1,1)] },
            'c_blocks': { 0: 1, 1: 0, 2: 0, 3: 0 }
        }
    ]
    
    U = [[0] * 110 for _ in range(25)]
    V = [[0] * 110 for _ in range(25)]
    W = [[0] * 110 for _ in range(25)]
    
    col = 0
    for block_idx in range(7):
        bd = block_defs[block_idx]
        for local_term_idx in range(7):
            u_loc, v_loc, w_loc = strassen_2x2[local_term_idx]
            
            # Expand U and V to 4x4 (16 entries)
            u_4x4 = [0] * 16
            v_4x4 = [0] * 16
            
            # u_loc is 4-dim: [a,b,c,d] corresponding to local (0,0), (0,1), (1,0), (1,1)
            local_indices = [(0,0), (0,1), (1,0), (1,1)]
            for l_idx, (lu, lv) in enumerate(local_indices):
                coeff_u = u_loc[l_idx]
                coeff_v = v_loc[l_idx]
                
                # Add to u_4x4
                for (gi, gj, g_coeff) in bd['x'][(lu, lv)]:
                    u_4x4[gi * 4 + gj] += coeff_u * g_coeff
                    
                # Add to v_4x4
                for (gi, gj, g_coeff) in bd['y'][(lu, lv)]:
                    v_4x4[gi * 4 + gj] += coeff_v * g_coeff
            
            # Expand W to 4x4 (16 entries)
            w_4x4 = [0] * 16
            # w_loc is 4-dim: [C00, C01, C10, C11]
            c_block_coeffs = bd['c_blocks']
            for c_idx in range(4):
                coeff_w = w_loc[c_idx]
                block_coeff = c_block_coeffs[c_idx]
                if block_coeff != 0:
                    # Map c_idx (0:00, 1:01, 2:10, 3:11) to block
                    bi = c_idx // 2
                    bj = c_idx % 2
                    for lu in range(2):
                        for lv in range(2):
                            gi = bi * 2 + lu
                            gj = bj * 2 + lv
                            w_4x4[gi * 4 + gj] += coeff_w * block_coeff
            
            # Copy to global U, V, W (5x5 = 25 entries)
            # Indices 0-15 correspond to 4x4 core
            for idx in range(16):
                U[idx][col] = u_4x4[idx]
                V[idx][col] = v_4x4[idx]
                W[idx][col] = w_4x4[idx]
            
            col += 1
            
    # --- Part 2: Schoolbook for remaining terms ---
    # Terms where i=4 or j=4 or k=4
    # Total schoolbook terms = 125
    # Core terms (i<4, j<4, k<4) = 64
    # Remaining = 61
    
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if i < 4 and j < 4 and k < 4:
                    continue # Already handled by Strassen core
                
                row_u = i * 5 + j
                row_v = j * 5 + k
                row_w = k * 5 + i
                
                U[row_u][col] = 1
                V[row_v][col] = 1
                W[row_w][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
