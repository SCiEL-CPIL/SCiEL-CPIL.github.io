"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # We are constructing rank 77 decomposition
    # 49 from 4x4x4 Strassen
    # 28 from 4x4x2 (4 * 7)
    
    # Base 2x2x2 Strassen terms
    # u, v, w are length 4
    # Indices: 00, 01, 10, 11
    strassen_2x2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        U_terms.append(u_vec)
        V_terms.append(v_vec)
        W_terms.append(w_vec)

    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    # Blocks: 00, 01, 10, 11
    # U_s, V_s, W_s are coefficients for the 4 blocks
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    U1 = [1, 0, 0, 1]
    V1 = [1, 0, 0, 1]
    W1 = [1, 0, 0, 1]
    
    # M2 = (A10+A11)B00 -> C10, C11(-)
    U2 = [0, 0, 1, 1]
    V2 = [1, 0, 0, 0]
    W2 = [0, 0, 1, -1]
    
    # M3 = A00(B01-B11) -> C01, C11
    U3 = [1, 0, 0, 0]
    V3 = [0, 1, 0, -1]
    W3 = [0, 1, 0, 1]
    
    # M4 = A11(B10-B00) -> C00, C10
    U4 = [0, 0, 0, 1]
    V4 = [0, 0, 1, 0] # Wait, B10 is block 10 (index 2), B00 is block 00 (index 0).
    # V4 should be coeff for B blocks.
    # M4 = A11 * (B10 - B00).
    # So V4 = [-1, 0, 1, 0].
    V4 = [-1, 0, 1, 0]
    W4 = [1, 0, 1, 0]
    
    # M5 = (A00+A01)B11 -> C00(-), C01
    U5 = [1, 1, 0, 0]
    V5 = [0, 0, 0, 1]
    W5 = [-1, 1, 0, 0]
    
    # M6 = (A10-A00)(B00+B01) -> C11
    U6 = [-1, 0, 1, 0]
    V6 = [1, 1, 0, 0]
    W6 = [0, 0, 0, 1]
    
    # M7 = (A01-A11)(B10+B11) -> C00
    U7 = [0, 1, 0, -1]
    V7 = [0, 0, 1, 1]
    W7 = [1, 0, 0, 0]
    
    strassen_4x4x4_blocks = [
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6),
        (U7, V7, W7)
    ]
    
    for U_b, V_b, W_b in strassen_4x4x4_blocks:
        for u_base, v_base, w_base in strassen_2x2x2:
            # Expand to length 16
            u_full = [0] * 16
            v_full = [0] * 16
            w_full = [0] * 16
            
            for block_idx in range(4):
                c_u = U_b[block_idx]
                c_v = V_b[block_idx]
                c_w = W_b[block_idx]
                
                start = block_idx * 4
                for j in range(4):
                    u_full[start + j] = c_u * u_base[j]
                    v_full[start + j] = c_v * v_base[j]
                    w_full[start + j] = c_w * w_base[j]
                    
            add_term(u_full, v_full, w_full)

    # --- Part 2: 4x4x2 (Rank 28) ---
    # A is 4x4 (4 blocks of 2x2)
    # B is 4x2 (2 blocks of 2x2: Top, Bot)
    # C is 4x2 (2 blocks of 2x2: Top, Bot)
    
    # Terms:
    # 1. A00 * B_top -> C_top
    # 2. A01 * B_bot -> C_top
    # 3. A10 * B_top -> C_bot
    # 4. A11 * B_bot -> C_bot
    
    # U coeffs for A blocks (len 4)
    # V coeffs for B blocks (len 2)
    # W coeffs for C blocks (len 2)
    
    part2_terms = [
        ([1, 0, 0, 0], [1, 0], [1, 0]), # A00 Btop -> Ctop
        ([0, 1, 0, 0], [0, 1], [1, 0]), # A01 Bbot -> Ctop
        ([0, 0, 1, 0], [1, 0], [0, 1]), # A10 Btop -> Cbot
        ([0, 0, 0, 1], [0, 1], [0, 1])  # A11 Bbot -> Cbot
    ]
    
    for U_b, V_b, W_b in part2_terms:
        for u_base, v_base, w_base in strassen_2x2x2:
            u_full = [0] * 16
            v_full = [0] * 8
            w_full = [0] * 8
            
            # A has 4 blocks of size 4
            for block_idx in range(4):
                c_u = U_b[block_idx]
                start = block_idx * 4
                for j in range(4):
                    u_full[start + j] = c_u * u_base[j]
                    
            # B has 2 blocks of size 4 (Top, Bot)
            for block_idx in range(2):
                c_v = V_b[block_idx]
                start = block_idx * 4
                for j in range(4):
                    v_full[start + j] = c_v * v_base[j]
                    
            # C has 2 blocks of size 4 (Top, Bot)
            for block_idx in range(2):
                c_w = W_b[block_idx]
                start = block_idx * 4
                for j in range(4):
                    w_full[start + j] = c_w * w_base[j]
                    
            add_term(u_full, v_full, w_full)

    # Convert to lists of lists for output
    U = [list(map(int, term)) for term in U_terms]
    V = [list(map(int, term)) for term in V_terms]
    W = [list(map(int, term)) for term in W_terms]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
