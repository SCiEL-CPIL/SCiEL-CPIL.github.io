import sys
import time

def construct():
    """
    Rank-50 tensor decomposition for 2x4 by 4x7 matrix multiplication.
    
    The decomposition splits the middle dimension (size 4) into two halves (size 2).
    For each output column block, we perform matrix multiplication using these halves.
    Output columns 0,1; 2,3; 4,5 are handled in pairs.
    Output column 6 is handled individually.
    
    For a pair of output columns (2 cols), we have two sub-problems:
    1. A_left (2x2) * B_top (2x2) -> C_block (2x2)
    2. A_right (2x2) * B_bottom (2x2) -> C_block (2x2)
    Each sub-problem uses Strassen's algorithm (Rank 7).
    Total per pair: 14.
    3 pairs: 42.
    
    For the single output column (1 col), we have two sub-problems:
    1. A_left (2x2) * B_top (2x1) -> C_block (2x1)
    2. A_right (2x2) * B_bottom (2x1) -> C_block (2x1)
    Each sub-problem uses Schoolbook (Rank 4).
    Total per single: 8.
    
    Total Rank: 42 + 8 = 50.
    """
    n, m, p = 2, 4, 7
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Indices for A (2x4)
    # A_L: cols 0,1. A_R: cols 2,3
    A_L = [(0, 0), (0, 1), (1, 0), (1, 1)]
    A_R = [(0, 2), (0, 3), (1, 2), (1, 3)]
    
    # Indices for B (4x7)
    # B_top: rows 0,1. B_bot: rows 2,3
    
    def add_strassen(A_indices, B_indices, C_indices):
        nonlocal col
        # Strassen 2x2x2
        # A, B, C are 4-element lists of (i,j) indices
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        def proc(A_c, B_c, C_c):
            nonlocal col
            for la, c in A_c.items():
                gi, gj = A_indices[la]
                U[gi*m + gj][col] += c
            for lb, c in B_c.items():
                gi, gj = B_indices[lb]
                V[gi*p + gj][col] += c
            for lc, c in C_c.items():
                gi, gj = C_indices[lc]
                W[gj*n + gi][col] += c
            col += 1
        
        # Map local 0..3 to global indices
        # A: 00, 01, 10, 11
        # B: 00, 01, 10, 11
        # C: 00, 01, 10, 11
        
        proc({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}) # M1
        proc({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}) # M2
        proc({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}) # M3
        proc({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}) # M4
        proc({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}) # M5
        proc({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}) # M6
        proc({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}) # M7

    def add_schoolbook_221(A_indices, B_indices, C_indices):
        nonlocal col
        # A is 2x2, B is 2x1, C is 2x1
        # c0 = a00*b0 + a01*b1
        # c1 = a10*b0 + a11*b1
        def proc_mult(A_idx, B_idx, C_idx, coeff=1):
            nonlocal col
            gi, gj = A_indices[A_idx]
            U[gi*m + gj][col] += coeff
            gi, gj = B_indices[B_idx]
            V[gi*p + gj][col] += coeff
            gi, gj = C_indices[C_idx]
            W[gj*n + gi][col] += coeff
            col += 1
            
        proc_mult(0, 0, 0, 1) # a00*b0
        proc_mult(1, 1, 0, 1) # a01*b1
        proc_mult(2, 0, 1, 1) # a10*b0
        proc_mult(3, 1, 1, 1) # a11*b1

    # Process 3 pairs of columns
    for pair_idx in range(3):
        j_start = pair_idx * 2
        j_end = j_start + 2
        
        # B_top for these cols: rows 0,1
        B_top = [(0, j_start), (0, j_end-1), (1, j_start), (1, j_end-1)]
        # B_bot for these cols: rows 2,3
        B_bot = [(2, j_start), (2, j_end-1), (3, j_start), (3, j_end-1)]
        
        # C for these cols
        C = [(0, j_start), (0, j_end-1), (1, j_start), (1, j_end-1)]
        
        add_strassen(A_L, B_top, C)
        add_strassen(A_R, B_bot, C)

    # Process single column
    j_single = 6
    B_top_single = [(0, j_single), (1, j_single)]
    B_bot_single = [(2, j_single), (3, j_single)]
    C_single = [(0, j_single), (1, j_single)]
    
    add_schoolbook_221(A_L, B_top_single, C_single)
    add_schoolbook_221(A_R, B_bot_single, C_single)
    
    return {"U": U, "V": V, "W": W}
