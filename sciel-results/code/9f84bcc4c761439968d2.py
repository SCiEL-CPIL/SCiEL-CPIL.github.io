import time
import numpy as np

def construct():
    """
    Rank-42 exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.
    
    Strategy: Split the 4-dimension middle into two 2-dim blocks (top and bottom).
    C = A_top @ B_top + A_bottom @ B_bottom
    Each 2x2 @ 2x6 is decomposed into three 2x2 @ 2x2 sub-blocks using Strassen (rank 7 each).
    Total rank = 2 * 3 * 7 = 42.
    
    U: (n*m, rank) = (8, 42)
    V: (m*p, rank) = (24, 42)
    W: (p*n, rank) = (12, 42)
    
    W index convention: W[k*n+i, r] where k is column index (0..5), i is row index (0..1)
    So W index = k * n + i = k * 2 + i for output C[i][k]
    """
    n, m, p = 2, 4, 6
    rank_total = 42
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Strassen 2x2 @ 2x2 rank-7 decomposition
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a10+a11)*b00
    # P3 = a00*(b01-b11)
    # P4 = a11*(b10-b00)
    # P5 = (a00+a01)*b11
    # P6 = (a10-a00)*(b00+b01)
    # P7 = (a01-a11)*(b10+b11)
    #
    # c00 = P1 + P4 - P5 + P7
    # c01 = P3 + P5
    # c10 = P2 + P4
    # c11 = P1 - P2 + P3 + P6
    
    # For each product, store which local elements contribute with what sign
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_U = [
        {0: 1, 3: 1},       # P1: a00+a11
        {2: 1, 3: 1},       # P2: a10+a11
        {0: 1},             # P3: a00
        {3: 1},             # P4: a11
        {0: 1, 1: 1},       # P5: a00+a01
        {2: 1, 0: -1},      # P6: a10-a00
        {1: 1, 3: -1},      # P7: a01-a11
    ]
    strassen_V = [
        {0: 1, 3: 1},       # P1: b00+b11
        {0: 1},             # P2: b00
        {1: 1, 3: -1},      # P3: b01-b11
        {2: 1, 0: -1},      # P4: b10-b00
        {3: 1},             # P5: b11
        {0: 1, 1: 1},       # P6: b00+b01
        {2: 1, 3: 1},       # P7: b10+b11
    ]
    strassen_W = [
        {0: 1, 3: 1},       # P1: c00+c11
        {2: 1, 3: -1},      # P2: c10-c11
        {1: 1, 3: 1},       # P3: c01+c11
        {0: 1, 2: 1},       # P4: c00+c10
        {0: -1, 1: 1},      # P5: -c00+c01
        {3: 1},             # P6: c11
        {0: 1},             # P7: c00
    ]
    
    col_offset = 0
    
    # Iterate over the 3 column blocks of B (each 2 columns wide)
    # and the 2 row blocks of B (top 2 rows, bottom 2 rows)
    for j_block in range(3):  # column block of B: cols 2*j_block to 2*j_block+1
        for side in ['L', 'R']:  # L = top half of B (rows 0-1), R = bottom half (rows 2-3)
            if side == 'L':
                u_col_offset = 0      # A columns 0-1
                v_row_offset = 0      # B rows 0-1
            else:
                u_col_offset = 2      # A columns 2-3
                v_row_offset = 2      # B rows 2-3
            
            for prod_idx in range(7):
                col = col_offset + prod_idx
                
                # U local index mapping:
                # local 0 -> A[0][u_col_offset+0], U idx = 0*m + (u_col_offset+0) = u_col_offset
                # local 1 -> A[0][u_col_offset+1], U idx = 0*m + (u_col_offset+1) = u_col_offset+1
                # local 2 -> A[1][u_col_offset+0], U idx = 1*m + (u_col_offset+0) = 4+u_col_offset
                # local 3 -> A[1][u_col_offset+1], U idx = 1*m + (u_col_offset+1) = 4+u_col_offset+1
                local_U = {
                    0: 0 * m + (u_col_offset + 0),
                    1: 0 * m + (u_col_offset + 1),
                    2: 1 * m + (u_col_offset + 0),
                    3: 1 * m + (u_col_offset + 1),
                }
                for li, s in strassen_U[prod_idx].items():
                    U[local_U[li]][col] += s
                
                # V local index mapping:
                # local 0 -> B[v_row_offset+0][2*j_block+0], V idx = (v_row_offset+0)*p + (2*j_block+0)
                # local 1 -> B[v_row_offset+0][2*j_block+1], V idx = (v_row_offset+0)*p + (2*j_block+1)
                # local 2 -> B[v_row_offset+1][2*j_block+0], V idx = (v_row_offset+1)*p + (2*j_block+0)
                # local 3 -> B[v_row_offset+1][2*j_block+1], V idx = (v_row_offset+1)*p + (2*j_block+1)
                local_V = {
                    0: (v_row_offset + 0) * p + (2 * j_block + 0),
                    1: (v_row_offset + 0) * p + (2 * j_block + 1),
                    2: (v_row_offset + 1) * p + (2 * j_block + 0),
                    3: (v_row_offset + 1) * p + (2 * j_block + 1),
                }
                for li, s in strassen_V[prod_idx].items():
                    V[local_V[li]][col] += s
                
                # W local index mapping:
                # Output C[i][k] where i in 0..1, k in 2*j_block..2*j_block+1
                # W index convention: W[k*n+i, r] = W[k*2+i, r]
                # local 0 -> C[0][2*j_block+0], W idx = (2*j_block+0)*2 + 0
                # local 1 -> C[0][2*j_block+1], W idx = (2*j_block+1)*2 + 0
                # local 2 -> C[1][2*j_block+0], W idx = (2*j_block+0)*2 + 1
                # local 3 -> C[1][2*j_block+1], W idx = (2*j_block+1)*2 + 1
                local_W = {
                    0: (2 * j_block + 0) * n + 0,
                    1: (2 * j_block + 1) * n + 0,
                    2: (2 * j_block + 0) * n + 1,
                    3: (2 * j_block + 1) * n + 1,
                }
                for li, s in strassen_W[prod_idx].items():
                    W[local_W[li]][col] += s
            
            col_offset += 7
    
    assert col_offset == rank_total, f"Expected rank {rank_total}, got {col_offset}"
    
    return {"U": U, "V": V, "W": W}
