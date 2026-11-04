import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_to(U, V, W, col, a_entries, b_entries, c_entries):
        for idx, coeff in a_entries:
            U[idx][col] += coeff
        for idx, coeff in b_entries:
            V[idx][col] += coeff
        for idx, coeff in c_entries:
            W[idx][col] += coeff

    # Strassen 2x2x2 terms definitions
    # A_entries: list of (local_i, local_j, coeff)
    # B_entries: list of (local_i, local_j, coeff)
    # C_entries: list of (local_i, local_j, coeff)
    
    strassen_terms = [
        # t=0
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # t=1
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # t=2
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,-1), (1,1,1)]),
        # t=3
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,-1), (1,1,1)]),
        # t=4
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,1), (0,1,1)]),
        # t=5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(0,1,1), (1,0,-1)]),
        # t=6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,-1), (1,0,1)])
    ]
    
    def map_block_to_global(block_r, block_c, offset_r, offset_c, is_A=False, is_B=False, is_C=False):
        # block_r, block_c: 0 or 1 (local block index)
        # offset_r, offset_c: global offset
        # Returns list of (global_index, coeff) for the 4 elements in the 2x2 block
        # Note: The caller provides the coeff for the whole block term.
        # We need to return the list of global indices.
        pass

    # Part 1: 4x4x4 Strassen (Rank 49)
    # Sub-problems s=0..6
    # Define block mappings for A, B, C for each s
    # A blocks: 00, 11, 21, 22, etc.
    # We define for each s, the list of (block_r, block_c, coeff) for A, B, C
    
    # s=0: (A00+A11)(B00+B11)
    # s=1: (A10+A11)B00  -> A blocks: 10, 11. B block: 00.
    # Note: Local 2x2 indices in Strassen terms are 0,1.
    # In the sub-problem, we map these local indices to specific 2x2 blocks of the 4x4 matrix.
    
    # Let's define the mapping for each s.
    # A_map[s]: list of (local_i, local_j, block_r, block_c, coeff)
    # Actually, it's easier to just say: for term t in s, 
    # the A part involves specific global 2x2 blocks.
    
    # Let's define the blocks for each s.
    # s=0: A: [00, 11], B: [00, 11], C: [00, 11]
    # s=1: A: [10, 11], B: [00], C: [10, 11]
    # s=2: A: [00], B: [01, 11], C: [01, 11]
    # s=3: A: [11], B: [10, 00], C: [00, 11]
    # s=4: A: [00, 01], B: [11], C: [00, 01]
    # s=5: A: [10, 00], B: [00, 01], C: [01, 10]
    # s=6: A: [01, 11], B: [10, 11], C: [00, 10]
    
    # Note: The C blocks in Strassen terms are determined by the term t, not s.
    # The s determines WHICH blocks of A and B are used.
    # The t determines the coefficients and WHICH blocks of C are updated.
    
    # Wait, the C blocks in the formula $C_{ij} = \sum M_k$ are fixed for the whole 4x4x4 problem.
    # The term t in s=0 contributes to C blocks 00 and 11.
    # The term t in s=1 contributes to C blocks 10 and 11.
    # This is consistent with the Strassen formulas.
    
    # So for each s, we have a fixed set of A blocks and B blocks.
    # For each t in s, we use the coefficients from strassen_terms[t] for A and B,
    # and the coefficients from strassen_terms[t] for C.
    
    # We need to map the local indices (0,0) etc in strassen_terms to the actual global blocks.
    # In strassen_terms[t], the A entries are like (0,0,1).
    # This means "Local A block at (0,0) with coeff 1".
    # We need to know which global block corresponds to Local (0,0) for this s.
    
    # Let's define block_offsets for each s.
    # A_blocks[s]: dict mapping local (r,c) to global block (R,C)
    
    A_block_map = {
        0: {(0,0): (0,0), (1,1): (1,1)}, # s=0: A00, A11
        1: {(1,0): (1,0), (1,1): (1,1)}, # s=1: A10, A11
        2: {(0,0): (0,0)},               # s=2: A00
        3: {(1,1): (1,1)},               # s=3: A11
        4: {(0,0): (0,0), (0,1): (0,1)}, # s=4: A00, A01
        5: {(1,0): (1,0), (0,0): (0,0)}, # s=5: A10, A00
        6: {(0,1): (0,1), (1,1): (1,1)}  # s=6: A01, A11
    }
    
    B_block_map = {
        0: {(0,0): (0,0), (1,1): (1,1)}, # s=0: B00, B11
        1: {(0,0): (0,0)},               # s=1: B00
        2: {(0,1): (0,1), (1,1): (1,1)}, # s=2: B01, B11
        3: {(1,0): (1,0), (0,0): (0,0)}, # s=3: B10, B00
        4: {(1,1): (1,1)},               # s=4: B11
        5: {(0,0): (0,0), (0,1): (0,1)}, # s=5: B00, B01
        6: {(1,0): (1,0), (1,1): (1,1)}  # s=6: B10, B11
    }
    
    col = 0
    for s in range(7):
        for t in range(7):
            a_terms, b_terms, c_terms = strassen_terms[t]
            
            a_entries = []
            b_entries = []
            c_entries = []
            
            # Map A
            for li, lj, coeff in a_terms:
                # Find global block for local (li, lj) in map s
                # The key in A_block_map[s] is (li, lj)?
                # Yes, because the Strassen term uses local indices 0,1.
                # We defined the map using those local indices.
                if (li, lj) in A_block_map[s]:
                    gr, gc = A_block_map[s][(li, lj)]
                    # Expand 2x2 block to global indices
                    for di in range(2):
                        for dj in range(2):
                            gi = gr * 2 + di
                            gj = gc * 2 + dj
                            global_idx = gi * m + gj
                            a_entries.append((global_idx, coeff))
            
            # Map B
            for li, lj, coeff in b_terms:
                if (li, lj) in B_block_map[s]:
                    gr, gc = B_block_map[s][(li, lj)]
                    for di in range(2):
                        for dj in range(2):
                            gj = gr * 2 + di
                            gk = gc * 2 + dj
                            global_idx = gj * p + gk
                            b_entries.append((global_idx, coeff))
            
            # Map C
            # C blocks are determined by t, but we need to map local C indices to global C indices.
            # In the 4x4x4 problem, the local C indices (0,0) etc correspond to global blocks (0,0) etc.
            # So local (li, lj) maps to global block (li, lj).
            for li, lj, coeff in c_terms:
                gr, gc = li, lj
                for di in range(2):
                    for dj in range(2):
                        gk = gr * 2 + di
                        gi = gc * 2 + dj
                        global_idx = gk * n + gi
                        c_entries.append((global_idx, coeff))
            
            add_to(U, V, W, col, a_entries, b_entries, c_entries)
            col += 1
            
    # Part 2: 4x4x2 (Rank 28)
    # 4 multiplications
    # Mult 0: A00 B00 -> C00 (where B00 is top block of B, C00 is top block of C)
    # Mult 1: A01 B10 -> C00
    # Mult 2: A10 B00 -> C10
    # Mult 3: A11 B10 -> C10
    
    # Note: B is 4x2. 
    # B "top" block (rows 0-1, cols 4-5) is B00 in this context?
    # Let's call B_top and B_bot.
    # B_top: rows 0-1, cols 4-5.
    # B_bot: rows 2-3, cols 4-5.
    
    # C "top" block (rows 0-1, cols 4-5) is C_top.
    # C "bot" block (rows 2-3, cols 4-5) is C_bot.
    
    # A blocks:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    mults = [
        # (A_block_r, A_block_c, B_block_r, B_block_c, C_block_r, C_block_c)
        (0, 0, 0, 0, 0, 0), # A00, B_top, C_top
        (0, 1, 1, 0, 0, 0), # A01, B_bot, C_top
        (1, 0, 0, 0, 1, 0), # A10, B_top, C_bot
        (1, 1, 1, 0, 1, 0)  # A11, B_bot, C_bot
    ]
    
    for a_r, a_c, b_r, b_c, c_r, c_c in mults:
        for t in range(7):
            a_terms, b_terms, c_terms = strassen_terms[t]
            
            a_entries = []
            b_entries = []
            c_entries = []
            
            # Map A
            # Local (0,0) in term -> Global block (a_r, a_c)
            # We need to map each local index in a_terms to the global block.
            # Since A_block is 2x2, local (li, lj) maps to global (a_r*2+li, a_c*2+lj)
            for li, lj, coeff in a_terms:
                gi = a_r * 2 + li
                gj = a_c * 2 + lj
                global_idx = gi * m + gj
                a_entries.append((global_idx, coeff))
            
            # Map B
            # B_block is 2x2.
            # Global rows: b_r*2 + li. Global cols: 4 + lj (since B is 4x2, cols are 4,5)
            for li, lj, coeff in b_terms:
                gj = b_r * 2 + li
                gk = 4 + lj
                global_idx = gj * p + gk
                b_entries.append((global_idx, coeff))
            
            # Map C
            # C_block is 2x2.
            # Global rows (k): c_r*2 + li. Global cols (i): c_c*2 + lj.
            # Note: C is 6x4. Rows are k (0-5), Cols are i (0-3).
            # C_top: rows 4-5? No.
            # C is 4x6. C_{i,k}.
            # W index: k*n + i.
            # C_top corresponds to i=0,1. k=4,5.
            # So global row in W (which is k) is 4+li?
            # Let's check C_block_r.
            # In mults, c_r is 0 or 1.
            # If c_r=0, it's C_top (i=0,1).
            # If c_r=1, it's C_bot (i=2,3).
            # Wait, my notation in mults was (c_r, c_c).
            # I used c_r for the "row" of the block in the 2x1 grid of C blocks?
            # C is 4x2.
            # C_top: i=0,1.
            # C_bot: i=2,3.
            # So c_r=0 -> i offset 0. c_r=1 -> i offset 2.
            # k is always 4,5.
            
            for li, lj, coeff in c_terms:
                gk = 4 + li  # k is 4,5
                gi = c_r * 2 + lj # i is 0,1 or 2,3
                global_idx = gk * n + gi
                c_entries.append((global_idx, coeff))
            
            add_to(U, V, W, col, a_entries, b_entries, c_entries)
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
