import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    """
    Constructs a rank-42 decomposition for 2x4 x 4x6 matrix multiplication.
    Uses 3-term Strassen-like blocks (standard 7-term) applied to tiled sub-matrices.
    """
    n, m, p = DIMENSIONS
    
    # Start timer
    start_time = time.time()
    
    # Rank calculation:
    # A (2x4) split into 2 blocks (2x2)
    # B (4x6) split into 2 rows, each into 3 blocks (2x2) -> 6 blocks total
    # Each block mult (2x2 x 2x2) uses 7 Strassen terms
    # Total rank = 2 * 3 * 7 = 42
    rank = 42
    
    # Initialize U, V, W as numpy arrays for faster manipulation, then convert to list
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Define Strassen coefficients for 2x2 block multiplication
    # Local indices for 2x2 matrix:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # U factors (First matrix A, 2x2)
    # s: term index
    # (row, col) in local 2x2
    U_defs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    # V factors (Second matrix B, 2x2)
    V_defs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    # W factors (Output C, 2x2)
    # R00 = P1 + P4 - P5 + P7
    # R01 = P3 + P5
    # R10 = P2 + P4
    # R11 = P1 - P2 + P3 + P6
    W_defs = [
        {(0, 0): 1, (1, 1): 1},  # P1
        {(1, 0): 1, (1, 1): -1}, # P2
        {(0, 1): 1, (1, 1): 1},  # P3
        {(0, 0): 1, (1, 0): 1},  # P4
        {(0, 0): -1, (0, 1): 1}, # P5
        {(1, 1): 1},             # P6
        {(0, 0): 1}              # P7
    ]
    
    col_idx = 0
    # Iterate over A blocks (2) and B column blocks (3)
    # A blocks: a in [0, 1] corresponds to columns 0-1 and 2-3 of A
    # B blocks: b in [0, 1, 2] corresponds to columns 0-1, 2-3, 4-5 of B (within each row)
    # Actually, B is 4x6. Rows 0-1 are top block, 2-3 are bottom block.
    # But in the block mult logic:
    # C = [A0, A1] * [B00 B01 B02; B10 B11 B12]
    # C00_block = A0 B00 + A1 B10
    # C01_block = A0 B01 + A1 B11
    # C02_block = A0 B02 + A1 B12
    
    # We iterate over the 6 resulting output blocks of size 2x2.
    # Output block (out_row, out_col) where out_row in [0,1], out_col in [0,1,2]
    # Actually, Output C is 2x6.
    # Row 0 of C corresponds to local row 0 of the 2x2 blocks.
    # Row 1 of C corresponds to local row 1.
    
    # We generate rank components for each (a, b) pair of input blocks contributing to each output column block.
    # Wait, the standard tiling:
    # For each output column block j (0, 1, 2):
    #   C_0j = A_0 B_0j + A_1 B_1j
    #   This requires 2 block multiplications: A_0 B_0j and A_1 B_1j.
    #   Each block mult has 7 terms.
    #   Total terms for column j: 14.
    #   Total for 3 columns: 42.
    
    for b in range(3): # Output column block index
        for a in range(2): # Input A block index (top/bottom rows of the 4x4 middle dimension? No, A is 2x4)
            # A is 2x4. Split into A_left (cols 0-1) and A_right (cols 2-3).
            # B is 4x6. Split rows into B_top (rows 0-1) and B_bottom (rows 2-3).
            # B_top is 2x6, split into 3 2x2 blocks B_00, B_01, B_02.
            # B_bottom is 2x6, split into 3 2x2 blocks B_10, B_11, B_12.
            
            # The contribution to C's column block b is:
            # A_left (2x2) * B_top[b] (2x2)  --> contributes to C[:, 2b:2b+2]
            # A_right (2x2) * B_bottom[b] (2x2) --> contributes to C[:, 2b:2b+2]
            
            # Wait, standard matrix mult:
            # A (2x4) * B (4x6)
            # A = [ A_0 (2x2)  A_1 (2x2) ]
            # B = [ B_00 (2x2) B_01 (2x2) B_02 (2x2) ]  (Top 2 rows)
            #     [ B_10 (2x2) B_11 (2x2) B_12 (2x2) ]  (Bottom 2 rows)
            
            # C[:, 2b:2b+2] = A_0 * B_0b + A_1 * B_1b
            
            # So for each b, we have two 2x2 mults:
            # 1. A_0 * B_0b
            # 2. A_1 * B_1b
            
            # Loop over the two A-blocks
            for a in range(2):
                for s in range(7):
                    # Update U
                    # A_block a: rows 0-1 of A, cols a*2 to a*2+1
                    # U matrix is 2x4.
                    # Local index (i, j) in 2x2 map to global (i, a*2 + j)
                    for (i_loc, j_loc), coeff in U_defs[s].items():
                        u_row = i_loc * m + (a * 2 + j_loc)
                        U[u_row, col_idx] += coeff
                    
                    # Update V
                    # B_block:
                    # If a == 0, we use Top B block (rows 0-1 of B), col b
                    # If a == 1, we use Bottom B block (rows 2-3 of B), col b
                    # B is 4x6.
                    # Top B rows: 0,1. Bottom B rows: 2,3.
                    # Cols: b*2 to b*2+1.
                    # V matrix is 4x6.
                    # Local (j_loc, k_loc) in 2x2.
                    # If a==0: global row j_loc, global col b*2 + k_loc
                    # If a==1: global row 2 + j_loc, global col b*2 + k_loc
                    base_row = a * 2
                    for (j_loc, k_loc), coeff in V_defs[s].items():
                        v_row = base_row + j_loc
                        v_col = b * 2 + k_loc
                        V[v_row * p + v_col, col_idx] += coeff
                    
                    # Update W
                    # Output C is 2x6.
                    # This term contributes to C[:, 2b:2b+2].
                    # Local (i_loc, k_loc) in 2x2.
                    # W matrix is 6x2 (k*n + i).
                    # global k = b*2 + k_loc
                    # global i = i_loc
                    for (i_loc, k_loc), coeff in W_defs[s].items():
                        w_col = i_loc
                        w_row = b * 2 + k_loc
                        W[w_row * n + w_col, col_idx] += coeff
                    
                    col_idx += 1
                    if col_idx >= rank:
                        break
                if col_idx >= rank: break
            if col_idx >= rank: break
        if col_idx >= rank: break

    # Time check
    elapsed = time.time() - start_time
    if elapsed > 110:
        # If running out of time, return current best (though construction is fast)
        pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
