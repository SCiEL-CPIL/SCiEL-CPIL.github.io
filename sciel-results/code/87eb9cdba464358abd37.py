import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Total rank estimate: 8 * 7 (Strassen 2x2x2) + 4 * 11 (2x2x3) = 56 + 44 = 100
    rank = 100
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a rank-1 term
    def add_term(i_u, val_u, i_v, val_v, i_w, val_w):
        nonlocal col_idx
        if col_idx >= rank:
            return
        U[i_u][col_idx] += val_u
        V[i_v][col_idx] += val_v
        W[i_w][col_idx] += val_w
        col_idx += 1

    # Strassen 2x2x2 decomposition
    # u: 4 elements (A00, A01, A10, A11)
    # v: 4 elements (B00, B01, B10, B11)
    # w: 4 elements (C00, C01, C10, C11)
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # P2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # P3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # P5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # P7
    ]
    
    def add_strassen_block(row_off, col_off, b_row_off, b_col_off):
        # A is 2x2 at (row_off, col_off)
        # B is 2x2 at (b_row_off, b_col_off)
        # C is 2x2 at (row_off, b_col_off)
        
        for u, v, w in strassen_terms:
            # Map local indices to global
            # u corresponds to A[i][j] -> global index i*m + j
            # v corresponds to B[j][k] -> global index j*p + k
            # w corresponds to C[i][k] -> global index k*n + i
            
            for li in range(2):
                for lj in range(2):
                    if u[li*2 + lj] != 0:
                        # Wait, u is flattened row-major?
                        # u[0]=A00, u[1]=A01, u[2]=A10, u[3]=A11
                        # My list above: [1, 0, 0, 1] means A00=1, A01=0, A10=0, A11=1.
                        # So index in list is li*2 + lj.
                        pass
            
            # It's easier to iterate over the 4 local entries of U
            for li in range(2):
                for lj in range(2):
                    val_u = u[li*2 + lj]
                    if val_u == 0: continue
                    g_i = row_off + li
                    g_j = col_off + lj
                    # Add to U at g_i*m + g_j
                    # We need to add this to the current column
                    U[g_i * m + g_j][col_idx] += val_u
            
            for lj in range(2):
                for lk in range(2):
                    val_v = v[lj*2 + lk]
                    if val_v == 0: continue
                    g_j = b_row_off + lj
                    g_k = b_col_off + lk
                    V[g_j * p + g_k][col_idx] += val_v
            
            for li in range(2):
                for lk in range(2):
                    val_w = w[li*2 + lk]
                    if val_w == 0: continue
                    g_i = row_off + li
                    g_k = b_col_off + lk
                    W[g_k * n + g_i][col_idx] += val_w
            
            col_idx += 1

    # Schoolbook 2x2x1 decomposition
    # A 2x2, B 2x1, C 2x1
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    schoolbook_221_terms = [
        # u (4), v (2), w (2)
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1]),
    ]
    
    def add_221_block(row_off, col_off, b_row_off, b_col_off):
        for u, v, w in schoolbook_221_terms:
            for li in range(2):
                for lj in range(2):
                    val_u = u[li*2 + lj]
                    if val_u == 0: continue
                    g_i = row_off + li
                    g_j = col_off + lj
                    U[g_i * m + g_j][col_idx] += val_u
            
            for lj in range(2):
                val_v = v[lj]
                if val_v == 0: continue
                g_j = b_row_off + lj
                g_k = b_col_off
                V[g_j * p + g_k][col_idx] += val_v
            
            for li in range(2):
                val_w = w[li]
                if val_w == 0: continue
                g_i = row_off + li
                g_k = b_col_off
                W[g_k * n + g_i][col_idx] += val_w
            
            col_idx += 1

    # Define blocks
    # I1 = 0, I2 = 2
    # J1 = 0, J2 = 2
    # K1 = 0, K2 = 2, K3 = 4
    
    # 8 Strassen blocks (2x2x2)
    # (row_off, col_off, b_row_off, b_col_off)
    strassen_blocks = [
        (0, 0, 0, 0), # I1 J1 K1
        (0, 2, 2, 0), # I1 J2 K1
        (0, 0, 0, 2), # I1 J1 K2
        (0, 2, 2, 2), # I1 J2 K2
        (2, 0, 0, 0), # I2 J1 K1
        (2, 2, 2, 0), # I2 J2 K1
        (2, 0, 0, 2), # I2 J1 K2
        (2, 2, 2, 2), # I2 J2 K2
    ]
    
    for block in strassen_blocks:
        add_strassen_block(*block)
        
    # 4 blocks of 2x2x3
    # Split K3 (cols 4,5,6) into K3a (4,5) and K3b (6)
    # Wait, my 2x2x3 decomposition assumes splitting 3 cols into 2 and 1.
    # The block is 2x2 (A) * 2x3 (B).
    # My add_221_block handles 2x2 * 2x1.
    # I need to split the 2x3 B into 2x2 and 2x1.
    # So for each 2x2x3 block, I will call add_strassen_block for the 2x2 part and add_221_block for the 2x1 part.
    
    # Block 9: I1 J1 K3 (cols 4,5,6)
    # A: I1 (rows 0,1), J1 (cols 0,1)
    # B: J1 (rows 0,1), K3 (cols 4,5,6)
    # Split B into B_a (rows 0,1, cols 4,5) and B_b (rows 0,1, col 6)
    add_strassen_block(0, 0, 0, 4) # I1 J1, B rows 0,1 cols 4,5 -> C cols 4,5
    add_221_block(0, 0, 0, 6)      # I1 J1, B rows 0,1 col 6 -> C col 6
    
    # Block 10: I1 J2 K3
    # A: I1 (rows 0,1), J2 (cols 2,3)
    # B: J2 (rows 2,3), K3 (cols 4,5,6)
    add_strassen_block(0, 2, 2, 4) # I1 J2, B rows 2,3 cols 4,5
    add_221_block(0, 2, 2, 6)      # I1 J2, B rows 2,3 col 6
    
    # Block 11: I2 J1 K3
    # A: I2 (rows 2,3), J1 (cols 0,1)
    # B: J1 (rows 0,1), K3 (cols 4,5,6)
    add_strassen_block(2, 0, 0, 4) # I2 J1, B rows 0,1 cols 4,5
    add_221_block(2, 0, 0, 6)      # I2 J1, B rows 0,1 col 6
    
    # Block 12: I2 J2 K3
    # A: I2 (rows 2,3), J2 (cols 2,3)
    # B: J2 (rows 2,3), K3 (cols 4,5,6)
    add_strassen_block(2, 2, 2, 4) # I2 J2, B rows 2,3 cols 4,5
    add_221_block(2, 2, 2, 6)      # I2 J2, B rows 2,3 col 6

    # Check if we filled all columns
    # col_idx should be 100
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
