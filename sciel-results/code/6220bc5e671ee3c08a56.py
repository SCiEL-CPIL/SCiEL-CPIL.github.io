"""Matrix multiplication 4x5x6 - Strassen-based decomposition."""
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 6
    
    # We'll build the decomposition by splitting the middle dimension (5) into 2+3.
    # A is 4x5, split into A1 (4x2) and A2 (4x3)
    # B is 5x6, split into B1 (2x6) and B2 (3x6)
    # C = A1*B1 + A2*B2
    # C is 4x6.
    # A1*B1: 4x2 * 2x6 = 4x6, rank 4*2*6 = 48
    # A2*B2: 4x3 * 3x6 = 4x6, rank 4*3*6 = 72
    # Total: 120. Same as schoolbook.
    
    # Better: use Strassen where possible.
    # Let's try a different approach: split the 4x5x6 problem.
    # 
    # Actually, let's use the known result that 4x4x4 has rank 49 (Strassen).
    # We can decompose 4x5x6 as:
    # Split A (4x5) into A_left (4x4) and A_right (4x1)
    # Split B (5x6) into B_top (4x6) and B_bottom (1x6)
    # Split B_top (4x6) into B_tl (4x4) and B_tr (4x2)
    #
    # C = A_left * B_top + A_right * B_bottom
    #   = A_left * (B_tl + B_tr) + A_right * B_bottom
    #   = A_left * B_tl + A_left * B_tr + A_right * B_bottom
    #
    # A_left * B_tl: 4x4 * 4x4 = 4x4, rank 49 (Strassen)
    # A_left * B_tr: 4x4 * 4x2 = 4x2, rank 4*4*2 = 32
    # A_right * B_bottom: 4x1 * 1x6 = 4x6, rank 4*1*6 = 24
    # Total: 49 + 32 + 24 = 105
    
    # Let's implement this.
    
    def strassen_4x4_terms():
        """Return the 49 rank-1 terms for 4x4x4 matrix multiplication using Strassen."""
        # First, get the 7 terms for 2x2x2 Strassen
        s2 = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])
        ]
        
        # For 4x4x4, we apply Strassen at the top level with 2x2 blocks of size 2x2.
        # Top-level Strassen 4x4:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each Mi is a 2x2 matrix multiplication, decomposed using the 7 s2 terms.
        # Total: 7 * 7 = 49 terms.
        
        # Define the top-level block definitions for each Mi.
        # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), 
        #           A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
        # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3),
        #           B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
        # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3),
        #           C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)
        
        # For each top-level term, define which A and B blocks are used and with what coefficients.
        # Then for each s2 term within that, map to global indices.
        
        # Top-level definitions: (a_blocks, b_blocks, c_blocks)
        # a_blocks: list of (block_row, block_col, coeff)
        # b_blocks: list of (block_row, block_col, coeff)
        # c_blocks: list of (block_row, block_col, coeff)
        
        top_terms = [
            # M1 = (A00+A11)(B00+B11)
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # M2 = (A10+A11)B00
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            # M3 = A00(B01-B11)
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            # M4 = A11(B10-B00)
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            # M5 = (A00+A01)B11
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            # M6 = (A10-A00)(B00+B01)
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            # M7 = (A01-A11)(B10+B11)
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,1,1)])
        ]
        
        terms_4x4 = []
        
        for a_blocks, b_blocks, c_blocks in top_terms:
            # For each s2 term, we need to combine the contributions.
            for s2_u, s2_v, s2_w in s2:
                # s2_u = [u00, u01, u10, u11] for the local 2x2 A matrix
                # s2_v = [v00, v01, v10, v11] for the local 2x2 B matrix
                # s2_w = [w00, w01, w10, w11] for the local 2x2 C matrix
                
                # Build global u, v, w vectors of length 16 (for 4x4 matrices)
                u_global = [0] * 16
                v_global = [0] * 16
                w_global = [0] * 16
                
                # Map local 2x2 indices to global 4x4 indices
                # Local (lr, lc) in block (br, bc) -> Global row = br*2+lr, col = bc*2+lc
                # Global index = row * 4 + col
                
                # For u: s2_u[lr*2+lc] is the coefficient for local A[lr][lc]
                # Local A[lr][lc] comes from the top-level A blocks.
                # For each top-level A block (br, bc, coeff), the local A[lr][lc] 
                # corresponds to global A[br*2+lr][bc*2+lc].
                
                for lr in range(2):
                    for lc in range(2):
                        s2_coeff = s2_u[lr*2+lc]
                        if s2_coeff != 0:
                            # This local entry (lr, lc) gets contributions from all top-level A blocks
                            # that map to this local position. But actually, each top-level term
                            # defines a specific combination of blocks.
                            # The local 2x2 matrix X has entries X[lr][lc] = sum of coeff * A[br*2+lr][bc*2+lc]
                            # for all (br, bc, coeff) in a_blocks where the local position matches.
                            # Wait, this is not quite right. The top-level term defines X as a sum of blocks.
                            # X = sum coeff * A_block
                            # So X[lr][lc] = sum coeff * A[br*2+lr][bc*2+lc] for each (br,bc,coeff) in a_blocks
                            
                            for br, bc, coeff in a_blocks:
                                global_row = br*2 + lr
                                global_col = bc*2 + lc
                                global_idx = global_row * 4 + global_col
                                u_global[global_idx] += s2_coeff * coeff
                
                # Similarly for v
                for lr in range(2):
                    for lc in range(2):
                        s2_coeff = s2_v[lr*2+lc]
                        if s2_coeff != 0:
                            for br, bc, coeff in b_blocks:
                                global_row = br*2 + lr
                                global_col = bc*2 + lc
                                global_idx = global_row * 4 + global_col
                                v_global[global_idx] += s2_coeff * coeff
                
                # Similarly for w
                for lr in range(2):
                    for lc in range(2):
                        s2_coeff = s2_w[lr*2+lc]
                        if s2_coeff != 0:
                            for br, bc, coeff in c_blocks:
                                global_row = br*2 + lr
                                global_col = bc*2 + lc
                                global_idx = global_row * 4 + global_col
                                w_global[global_idx] += s2_coeff * coeff
                
                # Check if this term is non-zero
                if any(x != 0 for x in u_global) and any(x != 0 for x in v_global) and any(x != 0 for x in w_global):
                    terms_4x4.append((u_global, v_global, w_global))
        
        return terms_4x4
    
    # Get the 49 terms for 4x4x4
    terms_4x4 = strassen_4x4_terms()
    
    # Now build the full 4x5x6 decomposition.
    # A (4x5) -> A_left (4x4, cols 0-3) + A_right (4x1, col 4)
    # B (5x6) -> B_top (4x6, rows 0-3) + B_bottom (1x6, row 4)
    # B_top (4x6) -> B_tl (4x4, cols 0-3) + B_tr (4x2, cols 4-5)
    
    # C = A_left * B_tl + A_left * B_tr + A_right * B_bottom
    
    # Part 1: A_left * B_tl (4x4 * 4x4 -> 4x4), using Strassen (49 terms)
    # Part 2: A_left * B_tr (4x4 * 4x2 -> 4x2), schoolbook (32 terms)
    # Part 3: A_right * B_bottom (4x1 * 1x6 -> 4x6), schoolbook (24 terms)
    
    total_rank = 49 + 32 + 24  # 105
    
    U = np.zeros((20, total_rank))  # 4*5 = 20
    V = np.zeros((30, total_rank))  # 5*6 = 30
    W = np.zeros((24, total_rank))  # 6*4 = 24
    
    col = 0
    
    # Part 1: A_left * B_tl using Strassen 4x4x4
    for u4, v4, w4 in terms_4x4:
        # u4 is length 16, corresponding to A_left (4x4), indexed by row*4+col
        # v4 is length 16, corresponding to B_tl (4x4), indexed by row*4+col
        # w4 is length 16, corresponding to C_left (4x4), indexed by row*4+col
        
        for idx in range(16):
            if u4[idx] != 0:
                # A_left row = idx // 4, col = idx % 4
                # Global A index: row * 5 + col (since A is 4x5)
                a_row = idx // 4
                a_col = idx % 4
                U[a_row * 5 + a_col, col] = u4[idx]
            
            if v4[idx] != 0:
                # B_tl row = idx // 4, col = idx % 4
                # Global B index: row * 6 + col (since B is 5x6)
                b_row = idx // 4
                b_col = idx % 4
                V[b_row * 6 + b_col, col] = v4[idx]
            
            if w4[idx] != 0:
                # C_left row = idx // 4, col = idx % 4
                # Global C index: row * 4 + col (since C is 4x6, but this part is 4x4)
                c_row = idx // 4
                c_col = idx % 4
                W[c_row * 4 + c_col, col] = w4[idx]
        
        col += 1
    
    # Part 2: A_left * B_tr (4x4 * 4x2 -> 4x2), schoolbook
    # A_left: rows 0-3, cols 0-3 of A
    # B_tr: rows 0-3, cols 4-5 of B
    # C_mid: rows 0-3, cols 4-5 of C
    for i in range(4):
        for j in range(4):
            for k in range(2):
                # A[i][j] * B[j][4+k] -> C[i][4+k]
                U[i * 5 + j, col] = 1
                V[j * 6 + (4 + k), col] = 1
                W[(4 + k) * 4 + i, col] = 1
                col += 1
    
    # Part 3: A_right * B_bottom (4x1 * 1x6 -> 4x6), schoolbook
    # A_right: rows 0-3, col 4 of A
    # B_bottom: row 4, cols 0-5 of B
    # C_right: rows 0-3, cols 0-5 of C
    for i in range(4):
        for k in range(6):
            # A[i][4] * B[4][k] -> C[i][k]
            U[i * 5 + 4, col] = 1
            V[4 * 6 + k, col] = 1
            W[k * 4 + i, col] = 1
            col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
