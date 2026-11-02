"""Rank-77 exact tensor decomposition for 4x4 * 4x6 matrix multiplication using mixed Strassen/Schoolbook."""

import time

def construct():
    """
    Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: 
    1. Outer: Strassen (rank 7) on 2x2 blocks of A (4x4) and B (4x6 treated as 2x3 blocks).
    2. Inner: For each outer term, decompose the 2x2 * 2x3 product.
       - Split 2x3 into 2x2 (cols 0-1) and 2x1 (col 2).
       - Use Strassen (rank 7) for 2x2 * 2x2 part.
       - Use Schoolbook (rank 4) for 2x2 * 2x1 part.
       - Total inner rank: 7 + 4 = 11.
    3. Total rank: 7 * 11 = 77.
    """
    start_time = time.time()
    DEADLINE = 150.0  # seconds
    
    n, m, p = 4, 4, 6
    
    rank = 77
    U = [[0] * rank for _ in range(n * m)]  # 16 x 77
    V = [[0] * rank for _ in range(m * p)]  # 24 x 77
    W = [[0] * rank for _ in range(p * n)]  # 24 x 77
    
    # Outer Strassen steps
    # A blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) (each 2x2)
    # B blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) (each 2x3)
    # C blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) (each 2x3)
    
    # Coefficients for A blocks, B blocks, and C block contributions
    # Format: { 'A': [(block_id, coeff), ...], 'B': [(block_id, coeff), ...], 'C': [(block_id, coeff), ...] }
    outer_steps = [
        # M1 = (A00 + A11)(B00 + B11)
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (1, -1), (2, -1), (3, 1)]},
        # M2 = (A10 + A11)B00
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, 1)]},
        # M3 = A00(B01 - B11)
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, -1)]},
        # M4 = A11(B10 - B00)
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, -1), (2, 1)]},
        # M5 = (A00 + A01)B11
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, 1), (1, 1)]},
        # M6 = (A10 - A00)(B00 + B01)
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(0, -1), (3, -1)]},
        # M7 = (A01 - A11)(B10 + B11)
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1), (1, -1)]}
    ]
    
    # Inner Strassen steps for 2x2 * 2x2
    # Local indices: A (2x2), B (2x2), C (2x2)
    inner_strassen = [
        # 1: (A00+A11)(B00+B11) -> C00, C11
        {'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)], 'C': [(0,0,1), (1,1,1)]},
        # 2: (A10+A11)B00 -> C10, C11
        {'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1), (1,1,-1)]},
        # 3: A00(B01-B11) -> C01, C11
        {'A': [(0,0,1)], 'B': [(0,1,1), (1,1,-1)], 'C': [(0,1,1), (1,1,1)]},
        # 4: A11(B10-B00) -> C00, C10
        {'A': [(1,1,1)], 'B': [(1,0,1), (0,0,-1)], 'C': [(0,0,1), (1,0,1)]},
        # 5: (A00+A01)B11 -> C00, C01
        {'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1), (0,1,1)]},
        # 6: (A10-A00)(B00+B01) -> C11
        {'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)], 'C': [(1,1,1)]},
        # 7: (A01-A11)(B10+B11) -> C00
        {'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)], 'C': [(0,0,1)]}
    ]
    
    # Inner Schoolbook steps for 2x2 * 2x1 -> 2x1
    # A (2x2), B (2x1), C (2x1)
    inner_schoolbook = [
        # C0 = A00 B0 + A01 B1
        {'A': [(0,0,1)], 'B': [(0,0,1)], 'C': [(0,0,1)]},
        {'A': [(0,1,1)], 'B': [(1,0,1)], 'C': [(0,0,1)]},
        # C1 = A10 B0 + A11 B1
        {'A': [(1,0,1)], 'B': [(0,0,1)], 'C': [(1,0,1)]},
        {'A': [(1,1,1)], 'B': [(1,0,1)], 'C': [(1,0,1)]}
    ]
    
    # Helper to map local block indices to global indices
    def a_block_to_global(block_id, r, s):
        if block_id == 0: return r * 4 + s
        elif block_id == 1: return r * 4 + (s + 2)
        elif block_id == 2: return (r + 2) * 4 + s
        elif block_id == 3: return (r + 2) * 4 + (s + 2)
        return -1

    def b_block_to_global(block_id, r, c):
        if block_id == 0: return r * 6 + c
        elif block_id == 1: return r * 6 + (c + 3)
        elif block_id == 2: return (r + 2) * 6 + c
        elif block_id == 3: return (r + 2) * 6 + (c + 3)
        return -1

    def c_block_to_global(block_id, r, c):
        # W index is K*4 + I, where K is col, I is row
        if block_id == 0: return c * 4 + r
        elif block_id == 1: return (c + 3) * 4 + r
        elif block_id == 2: return c * 4 + (r + 2)
        elif block_id == 3: return (c + 3) * 4 + (r + 2)
        return -1

    term_idx = 0
    
    for step in outer_steps:
        # Extract coefficients for A and B blocks
        # A_coeffs[block_id] = coeff
        # B_coeffs[block_id] = coeff
        A_coeffs = {0: 0, 1: 0, 2: 0, 3: 0}
        for bid, coeff in step['A']:
            A_coeffs[bid] = coeff
            
        B_coeffs = {0: 0, 1: 0, 2: 0, 3: 0}
        for bid, coeff in step['B']:
            B_coeffs[bid] = coeff

        # C coefficients: block_id -> coeff
        C_coeffs = {0: 0, 1: 0, 2: 0, 3: 0}
        for bid, coeff in step['C']:
            C_coeffs[bid] = coeff

        # Process Inner Strassen (for cols 0,1 of B)
        for s_step in inner_strassen:
            # Accumulate global U, V, W contributions
            # U: sum over A local (r,s) and outer block bid
            # V: sum over B local (r,c) and outer block bid
            # W: sum over C local (r,c) and outer block bid
            
            # For U:
            # Local A term (r,s) has coeff a_ls.
            # Global A index for block bid is a_block_to_global(bid, r, s).
            # Contribution to U[global_a_idx][term_idx] += a_ls * A_coeffs[bid]
            for r, s, a_coeff in s_step['A']:
                for bid in range(4):
                    if A_coeffs[bid] != 0:
                        g_idx = a_block_to_global(bid, r, s)
                        if g_idx >= 0:
                            U[g_idx][term_idx] += a_coeff * A_coeffs[bid]
            
            # For V:
            # Local B term (r,c) has coeff b_rc. Note: c is 0 or 1 (left part).
            # Global B index for block bid is b_block_to_global(bid, r, c).
            # Contribution to V[global_b_idx][term_idx] += b_rc * B_coeffs[bid]
            for r, c, b_coeff in s_step['B']:
                for bid in range(4):
                    if B_coeffs[bid] != 0:
                        g_idx = b_block_to_global(bid, r, c)
                        if g_idx >= 0:
                            V[g_idx][term_idx] += b_coeff * B_coeffs[bid]
            
            # For W:
            # Local C term (r,c) has coeff c_rc. Note: c is 0 or 1 (left part).
            # Global C index for block bid is c_block_to_global(bid, r, c).
            # Contribution to W[global_c_idx][term_idx] += c_rc * C_coeffs[bid]
            for r, c, c_coeff in s_step['C']:
                for bid in range(4):
                    if C_coeffs[bid] != 0:
                        g_idx = c_block_to_global(bid, r, c)
                        if g_idx >= 0:
                            W[g_idx][term_idx] += c_coeff * C_coeffs[bid]
            
            term_idx += 1
        
        # Process Inner Schoolbook (for col 2 of B)
        for s_step in inner_schoolbook:
            # For U:
            for r, s, a_coeff in s_step['A']:
                for bid in range(4):
                    if A_coeffs[bid] != 0:
                        g_idx = a_block_to_global(bid, r, s)
                        if g_idx >= 0:
                            U[g_idx][term_idx] += a_coeff * A_coeffs[bid]
            
            # For V:
            # Local B term (r, 0) corresponds to col 2 of the 2x3 block.
            # So global col is 2.
            for r, c, b_coeff in s_step['B']:
                # c is always 0 in schoolbook B (2x1)
                for bid in range(4):
                    if B_coeffs[bid] != 0:
                        g_idx = b_block_to_global(bid, r, 2)
                        if g_idx >= 0:
                            V[g_idx][term_idx] += b_coeff * B_coeffs[bid]
            
            # For W:
            # Local C term (r, 0) corresponds to col 2 of the 2x3 block.
            for r, c, c_coeff in s_step['C']:
                # c is always 0 in schoolbook C (2x1)
                for bid in range(4):
                    if C_coeffs[bid] != 0:
                        g_idx = c_block_to_global(bid, r, 2)
                        if g_idx >= 0:
                            W[g_idx][term_idx] += c_coeff * C_coeffs[bid]
            
            term_idx += 1
            
        # Time guard
        if time.time() - start_time > DEADLINE - 10:
            return {"U": U, "V": V, "W": W}
            
    return {"U": U, "V": V, "W": W}
