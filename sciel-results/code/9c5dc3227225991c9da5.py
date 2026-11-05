import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start = time.time()
    deadline = start + 140  # 20s safety margin
    
    n, m, p = 4, 4, 7
    
    # Strategy: Use Strassen for the 4x4 * 4x4 part (first 4 columns of B)
    # and schoolbook for the 4x4 * 4x3 part (last 3 columns of B)
    # Total rank: 49 (Strassen 4x4x4) + 48 (schoolbook 4x4x3) = 97
    
    rank = 97
    U = np.zeros((n * m, rank), dtype=float)  # 16 x 97
    V = np.zeros((m * p, rank), dtype=float)  # 28 x 97
    W = np.zeros((p * n, rank), dtype=float)  # 28 x 97
    
    # --- Strassen part: 4x4 * 4x4 (first 4 cols of B) ---
    # We'll use recursive Strassen: 4x4 = 7 * (2x2 * 2x2), and each 2x2*2x2 = 7 rank-1 terms
    # Total: 7 * 7 = 49 terms
    
    # Inner Strassen terms for 2x2*2x2:
    # For X = [[x0,x1],[x2,x3]], Y = [[y0,y1],[y2,y3]], Z = X*Y:
    # M0 = (x0+x3)*(y0+y3) -> z00:1, z11:1
    # M1 = (x2+x3)*y0 -> z10:1, z11:-1
    # M2 = x0*(y1-y3) -> z01:1, z11:1
    # M3 = x3*(y2-y0) -> z00:1, z10:1
    # M4 = (x0+x1)*y3 -> z00:-1, z01:1
    # M5 = (x2-x0)*(y0+y1) -> z11:1
    # M6 = (x1-x3)*(y2+y3) -> z00:1
    
    inner_terms = [
        # (U_entries, V_entries, W_entries)
        # Each is list of (position, coeff) where position is 0,1,2,3 for 2x2
        # 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        ([ (0,1.0), (3,1.0) ], [ (0,1.0), (3,1.0) ], [ (0,1.0), (3,1.0) ]),
        ([ (2,1.0), (3,1.0) ], [ (0,1.0) ], [ (2,1.0), (3,-1.0) ]),
        ([ (0,1.0) ], [ (1,1.0), (3,-1.0) ], [ (1,1.0), (3,1.0) ]),
        ([ (3,1.0) ], [ (2,1.0), (0,-1.0) ], [ (0,1.0), (2,1.0) ]),
        ([ (0,1.0), (1,1.0) ], [ (3,1.0) ], [ (0,-1.0), (1,1.0) ]),
        ([ (2,1.0), (0,-1.0) ], [ (0,1.0), (1,1.0) ], [ (3,1.0) ]),
        ([ (1,1.0), (3,-1.0) ], [ (2,1.0), (3,1.0) ], [ (0,1.0) ]),
    ]
    
    # Outer Strassen terms for 4x4*4x4 using 2x2 blocks:
    # A blocks: A00, A01, A10, A11 (indices 0,1,2,3)
    # B blocks: B00, B01, B10, B11 (indices 0,1,2,3)
    # C blocks: C00, C01, C10, C11 (indices 0,1,2,3)
    
    outer_terms = [
        # M0 = (A00+A11)*(B00+B11) -> C00:1, C11:1
        ([ (0,1.0), (3,1.0) ], [ (0,1.0), (3,1.0) ], [ (0,1.0), (3,1.0) ]),
        # M1 = (A10+A11)*B00 -> C10:1, C11:-1
        ([ (2,1.0), (3,1.0) ], [ (0,1.0) ], [ (2,1.0), (3,-1.0) ]),
        # M2 = A00*(B01-B11) -> C01:1, C11:1
        ([ (0,1.0) ], [ (1,1.0), (3,-1.0) ], [ (1,1.0), (3,1.0) ]),
        # M3 = A11*(B10-B00) -> C00:1, C10:1
        ([ (3,1.0) ], [ (2,1.0), (0,-1.0) ], [ (0,1.0), (2,1.0) ]),
        # M4 = (A00+A01)*B11 -> C00:-1, C01:1
        ([ (0,1.0), (1,1.0) ], [ (3,1.0) ], [ (0,-1.0), (1,1.0) ]),
        # M5 = (A10-A00)*(B00+B01) -> C11:1
        ([ (2,1.0), (0,-1.0) ], [ (0,1.0), (1,1.0) ], [ (3,1.0) ]),
        # M6 = (A01-A11)*(B10+B11) -> C00:1
        ([ (1,1.0), (3,-1.0) ], [ (2,1.0), (3,1.0) ], [ (0,1.0) ]),
    ]
    
    # Helper: get global index for A element given block and local position
    def a_global_idx(block, local):
        br = block // 2  # 0 or 1
        bc = block % 2   # 0 or 1
        lr = local // 2  # 0 or 1
        lc = local % 2   # 0 or 1
        row = br * 2 + lr
        col = bc * 2 + lc
        return row * m + col  # A is 4x4, so m=4
    
    def b_global_idx(block, local, col_offset=0):
        br = block // 2
        bc = block % 2
        lr = local // 2
        lc = local % 2
        row = br * 2 + lr
        col = bc * 2 + lc + col_offset
        return row * p + col  # B is 4x7, so p=7
    
    def c_global_idx(block, local):
        br = block // 2
        bc = block % 2
        lr = local // 2
        lc = local % 2
        row = br * 2 + lr
        col = bc * 2 + lc
        return col * n + row  # W[k*n+i] where k=col, i=row
    
    # Build the 49 Strassen terms for 4x4*4x4
    col_offset_strassen = 0  # First 4 columns of B
    
    term_idx = 0
    for outer_i, (a_blocks, b_blocks, c_blocks) in enumerate(outer_terms):
        for inner_i, (u_entries, v_entries, w_entries) in enumerate(inner_terms):
            # For each inner term, we need to map the 2x2 positions to global indices
            # The inner term operates on a specific 2x2 block from A and B
            # But actually, the outer Strassen combines blocks, so each "block" in the outer
            # is a 2x2 matrix, and the inner Strassen decomposes the 2x2*2x2 product.
            
            # For the outer term, A has contributions from certain blocks with coefficients
            # For each such block, the inner Strassen term has specific entries
            
            # Actually, let me reconsider. The outer Strassen term is:
            # M_outer = (linear combo of A blocks) * (linear combo of B blocks)
            # Each "linear combo of A blocks" is a 2x2 matrix (since each block is 2x2)
            # Similarly for B.
            # So M_outer = X * Y where X and Y are 2x2 matrices
            # Then inner Strassen decomposes X*Y into 7 rank-1 terms.
            
            # So for a given outer term:
            # X = sum of coeff_a * A_block for each (block, coeff) in a_blocks
            # Y = sum of coeff_b * B_block for each (block, coeff) in b_blocks
            
            # For the inner Strassen term inner_i, we have:
            # u_entries: which elements of X (i.e., which (br,bc,lr,lc) positions) and with what coeff
            # v_entries: which elements of Y
            # w_entries: which elements of Z (the result)
            
            # But X is a 2x2 matrix formed by combining A blocks.
            # Element (lr, lc) of X = sum over (block, coeff) in a_blocks of coeff * A_block[lr, lc]
            # where A_block[lr, lc] is the element at local position (lr, lc) of that block.
            
            # So for u_entries, for each (pos, coeff_u) in u_entries:
            #   pos corresponds to (lr, lc) in the 2x2 X matrix
            #   The contribution to U is: for each (block, coeff_a) in a_blocks:
            #     U[a_global_idx(block, pos)] += coeff_u * coeff_a
            
            # Similarly for V and W.
            
            for (pos_u, coeff_u) in u_entries:
                lr = pos_u // 2
                lc = pos_u % 2
                for (block_a, coeff_a) in a_blocks:
                    gidx = a_global_idx(block_a, pos_u)
                    U[gidx, term_idx] += coeff_u * coeff_a
            
            for (pos_v, coeff_v) in v_entries:
                lr = pos_v // 2
                lc = pos_v % 2
                for (block_b, coeff_b) in b_blocks:
                    gidx = b_global_idx(block_b, pos_v, col_offset_strassen)
                    V[gidx, term_idx] += coeff_v * coeff_b
            
            for (pos_w, coeff_w) in w_entries:
                lr = pos_w // 2
                lc = pos_w % 2
                for (block_c, coeff_c) in c_blocks:
                    gidx = c_global_idx(block_c, pos_w)
                    W[gidx, term_idx] += coeff_w * coeff_c
            
            term_idx += 1
    
    # --- Schoolbook part: 4x4 * 4x3 (last 3 cols of B) ---
    # For i in 0..3, j in 0..3, k in 3..6:
    # U[i*m+j, term] = 1
    # V[j*p+k, term] = 1  
    # W[k*n+i, term] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(3, p):  # k = 3, 4, 5, 6
                U[i * m + j, term_idx] = 1.0
                V[j * p + k, term_idx] = 1.0
                W[k * n + i, term_idx] = 1.0
                term_idx += 1
    
    # Verify term_idx == rank
    assert term_idx == rank, f"Expected rank {rank}, got {term_idx}"
    
    # Convert to lists for the required format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
