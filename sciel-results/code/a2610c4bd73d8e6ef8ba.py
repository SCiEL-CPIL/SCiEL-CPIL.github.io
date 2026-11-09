"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 using Strassen on 2x2 blocks
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block definitions
    # A blocks: (row_start, col_start)
    # 00: (0,0), 01: (0,2), 10: (2,0), 11: (2,2)
    blocks_A = [
        (0, 0), (0, 2),
        (2, 0), (2, 2)
    ]
    blocks_B = [
        (0, 0), (0, 2),
        (2, 0), (2, 2)
    ]
    
    # P1 = A00 B00, P2 = A01 B10, P3 = A00 B01, P4 = A01 B11
    # P5 = A10 B00, P6 = A11 B10, P7 = A10 B01, P8 = A11 B11
    # Mapping:
    # Block C00 = P1 + P2
    # Block C01 = P3 + P4
    # Block C10 = P5 + P6
    # Block C11 = P7 + P8
    
    # We need to sum the contributions.
    # Each Strassen multiplication is a separate column in U, V, W.
    
    col_idx = 0
    
    # Define the 8 block products
    # (A_block_idx, B_block_idx, C_block_idx)
    # C_block_idx determines where the result goes in W.
    # C00 is top-left, C01 top-right, C10 bottom-left, C11 bottom-right.
    
    products = [
        (0, 0, 0), # A00 * B00 -> C00
        (1, 2, 0), # A01 * B10 -> C00
        (0, 1, 1), # A00 * B01 -> C01
        (1, 3, 1), # A01 * B11 -> C01
        (2, 0, 2), # A10 * B00 -> C10
        (3, 2, 2), # A11 * B10 -> C10
        (2, 1, 3), # A10 * B01 -> C11
        (3, 3, 3), # A11 * B11 -> C11
    ]
    
    for A_idx, B_idx, C_idx in products:
        r_start_A, c_start_A = blocks_A[A_idx]
        r_start_B, c_start_B = blocks_B[B_idx]
        r_start_C, c_start_C = blocks_A[C_idx] # C blocks same geometry
        
        # Local indices for 2x2 block
        # A_local:
        # a = A[r_start_A, c_start_A]
        # b = A[r_start_A, c_start_A+1]
        # c = A[r_start_A+1, c_start_A]
        # d = A[r_start_A+1, c_start_A+1]
        
        # B_local:
        # e = B[r_start_B, c_start_B]
        # f = B[r_start_B, c_start_B+1]
        # g = B[r_start_B+1, c_start_B]
        # h = B[r_start_B+1, c_start_B+1]
        
        # U indices (A):
        idx_a = r_start_A * m + c_start_A
        idx_b = r_start_A * m + (c_start_A + 1)
        idx_c = (r_start_A + 1) * m + c_start_A
        idx_d = (r_start_A + 1) * m + (c_start_A + 1)
        
        # V indices (B):
        idx_e = r_start_B * p + c_start_B
        idx_f = r_start_B * p + (c_start_B + 1)
        idx_g = (r_start_B + 1) * p + c_start_B
        idx_h = (r_start_B + 1) * p + (c_start_B + 1)
        
        # W indices (C):
        # W index = col * n + row
        # C block starts at r_start_C, c_start_C
        # c11 (block) -> C[r_start_C, c_start_C] -> W[c_start_C * n + r_start_C]
        # c12 (block) -> C[r_start_C, c_start_C+1] -> W[(c_start_C+1) * n + r_start_C]
        # c21 (block) -> C[r_start_C+1, c_start_C] -> W[c_start_C * n + (r_start_C+1)]
        # c22 (block) -> C[r_start_C+1, c_start_C+1] -> W[(c_start_C+1) * n + (r_start_C+1)]
        
        w_c11 = c_start_C * n + r_start_C
        w_c12 = (c_start_C + 1) * n + r_start_C
        w_c21 = c_start_C * n + (r_start_C + 1)
        w_c22 = (c_start_C + 1) * n + (r_start_C + 1)
        
        # 7 Strassen multiplications
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # m1
        U[idx_a][col_idx] += 1
        U[idx_d][col_idx] += 1
        V[idx_e][col_idx] += 1
        V[idx_h][col_idx] += 1
        W[w_c11][col_idx] += 1
        W[w_c12][col_idx] += -1
        W[w_c21][col_idx] += 1
        W[w_c22][col_idx] += 1
        col_idx += 1
        
        # m2
        U[idx_c][col_idx] += 1
        U[idx_d][col_idx] += 1
        V[idx_e][col_idx] += 1
        W[w_c21][col_idx] += 1
        W[w_c22][col_idx] += -1
        col_idx += 1
        
        # m3
        U[idx_a][col_idx] += 1
        V[idx_f][col_idx] += 1
        V[idx_h][col_idx] += -1
        W[w_c12][col_idx] += 1
        W[w_c22][col_idx] += 1
        col_idx += 1
        
        # m4
        U[idx_d][col_idx] += 1
        V[idx_g][col_idx] += 1
        V[idx_e][col_idx] += -1
        W[w_c11][col_idx] += 1
        W[w_c21][col_idx] += 1
        col_idx += 1
        
        # m5
        U[idx_a][col_idx] += 1
        U[idx_b][col_idx] += 1
        V[idx_h][col_idx] += 1
        W[w_c11][col_idx] += -1
        W[w_c12][col_idx] += 1
        col_idx += 1
        
        # m6
        U[idx_c][col_idx] += 1
        U[idx_a][col_idx] += -1
        V[idx_e][col_idx] += 1
        V[idx_f][col_idx] += 1
        W[w_c22][col_idx] += 1
        col_idx += 1
        
        # m7
        U[idx_b][col_idx] += 1
        U[idx_d][col_idx] += -1
        V[idx_g][col_idx] += 1
        V[idx_h][col_idx] += 1
        W[w_c11][col_idx] += 1
        col_idx += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
