import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=6
    
    # Rank calculation:
    # Part 1: A[0:2, :] * B[:, 0:6] -> C[0:2, 0:6]
    # Split B into 3 blocks of width 2.
    # Each block: A[0:2, 0:4] * B[:, 2k:2k+2]
    # Split A into 2 blocks of width 2.
    # Each sub-block: A[0:2, 2j:2j+2] * B[2j:2j+2, 2k:2k+2]
    # This is a 2x2x2 multiplication, rank 7.
    # Number of such blocks: 3 (B cols) * 2 (A cols) = 6 blocks.
    # Total rank for Part 1: 6 * 7 = 42.
    
    # Part 2: A[2:3, :] * B[:, 0:6] -> C[2:3, 0:6]
    # A is 1x4, B is 4x6.
    # Rank is 1*4*6 = 24.
    
    total_rank = 42 + 24
    rank = total_rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # Local indices in 2x2 block
        # A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # B: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # C: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        
        # Global indices
        # A
        ua0 = a_r * m + a_c
        ua1 = a_r * m + a_c + 1
        ua2 = (a_r + 1) * m + a_c
        ua3 = (a_r + 1) * m + a_c + 1
        
        # B
        vb0 = b_r * p + b_c
        vb1 = b_r * p + b_c + 1
        vb2 = (b_r + 1) * p + b_c
        vb3 = (b_r + 1) * p + b_c + 1
        
        # C
        wc0 = c_r * n + c_c
        wc1 = c_r * n + c_c + 1
        wc2 = (c_r + 1) * n + c_c
        wc3 = (c_r + 1) * n + c_c + 1
        
        # Strassen 7 terms
        # 1. m1 = a00 * (b11 - b01) -> c00 + c11
        add_term(ua0, vb3, wc0, 1)
        add_term(ua0, vb1, wc0, -1)
        add_term(ua0, vb3, wc3, 1)
        add_term(ua0, vb1, wc3, -1)
        
        # 2. m2 = (a00 + a01) * b00 -> c00 + c01
        add_term(ua0, vb0, wc0, 1)
        add_term(ua1, vb0, wc0, 1)
        add_term(ua0, vb0, wc1, 1)
        add_term(ua1, vb0, wc1, 1)
        
        # 3. m3 = (a10 + a11) * b11 -> c10 + c11
        add_term(ua2, vb3, wc2, 1)
        add_term(ua3, vb3, wc2, 1)
        add_term(ua2, vb3, wc3, 1)
        add_term(ua3, vb3, wc3, 1)
        
        # 4. m4 = a11 * (b10 - b00) -> c00 + c11
        add_term(ua3, vb2, wc0, 1)
        add_term(ua3, vb0, wc0, -1)
        add_term(ua3, vb2, wc3, 1)
        add_term(ua3, vb0, wc3, -1)
        
        # 5. m5 = (a00 + a10) * (b00 + b11) -> c00 + c01 + c10 + c11
        add_term(ua0, vb0, wc0, 1)
        add_term(ua0, vb3, wc0, 1)
        add_term(ua2, vb0, wc0, 1)
        add_term(ua2, vb3, wc0, 1)
        
        add_term(ua0, vb0, wc1, 1)
        add_term(ua0, vb3, wc1, 1)
        add_term(ua2, vb0, wc1, 1)
        add_term(ua2, vb3, wc1, 1)
        
        add_term(ua0, vb0, wc2, 1)
        add_term(ua0, vb3, wc2, 1)
        add_term(ua2, vb0, wc2, 1)
        add_term(ua2, vb3, wc2, 1)
        
        add_term(ua0, vb0, wc3, 1)
        add_term(ua0, vb3, wc3, 1)
        add_term(ua2, vb0, wc3, 1)
        add_term(ua2, vb3, wc3, 1)
        
        # 6. m6 = (a01 - a11) * (b10 + b11) -> c11
        add_term(ua1, vb2, wc3, 1)
        add_term(ua1, vb3, wc3, 1)
        add_term(ua3, vb2, wc3, -1)
        add_term(ua3, vb3, wc3, -1)
        
        # 7. m7 = (a10 - a00) * (b00 + b01) -> c00 (-1) + c10
        add_term(ua2, vb0, wc0, -1)
        add_term(ua2, vb1, wc0, -1)
        add_term(ua0, vb0, wc0, 1)
        add_term(ua0, vb1, wc0, 1)
        
        add_term(ua2, vb0, wc2, 1)
        add_term(ua2, vb1, wc2, 1)
        add_term(ua0, vb0, wc2, -1)
        add_term(ua0, vb1, wc2, -1)

    # Part 1: A[0:2] * B
    # A rows 0,1
    # B cols 0..5
    # C rows 0,1
    for b_c in range(0, p, 2):
        for a_c in range(0, m, 2):
            add_strassen_block(0, a_c, a_c, b_c, 0, b_c)
            
    # Part 2: A[2:3] * B
    # A row 2
    # B rows 0..3, cols 0..5
    # C row 2, cols 0..5
    for j in range(m):
        for k in range(p):
            add_term(2 * m + j, j * p + k, k * n + 2, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
