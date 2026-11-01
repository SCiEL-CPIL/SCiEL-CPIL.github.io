import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a Strassen block
    # A_block is 2x2, indices in A are (i, j) where i in 0..1, j in offset..offset+1
    # B_block is 2x6, indices in B are (j, k) where j in offset..offset+1, k in 0..5
    # C_block is 2x6, indices in C are (i, k) where i in 0..1, k in 0..5
    
    def add_strassen_block(a_row_offset, b_row_offset, w_col_offset):
        # a_row_offset is 0 for A1, 2 for A2? No, A is 2x4.
        # A1 is cols 0,1. A2 is cols 2,3.
        # So a_row_offset is column offset in A.
        # b_row_offset is row offset in B. B1 is rows 0,1. B2 is rows 2,3.
        
        # Strassen factors for A
        # L1 = a+d, L2 = c+d, L3 = a, L4 = d, L5 = a+b, L6 = c-d, L7 = b-d
        # Indices in A: 
        # a = (0, a_row_offset), b = (0, a_row_offset+1)
        # c = (1, a_row_offset), d = (1, a_row_offset+1)
        
        # Strassen factors for B (3 blocks of 2x2)
        # B is 2x6. Rows b_row_offset, b_row_offset+1. Cols 0..5.
        # Block 1: cols 0,1. x=(b_row_offset, 0), y=(b_row_offset, 1), z=(b_row_offset+1, 0), w=(b_row_offset+1, 1)
        # Block 2: cols 2,3.
        # Block 3: cols 4,5.
        
        for block_idx in range(3):
            col_start = block_idx * 2
            # x, y, z, w indices in B
            x_idx = b_row_offset * p + col_start
            y_idx = b_row_offset * p + col_start + 1
            z_idx = (b_row_offset + 1) * p + col_start
            w_idx = (b_row_offset + 1) * p + col_start + 1
            
            # 7 products
            # P1 = L1 * M1
            # P2 = L2 * M2
            # P3 = L3 * M3
            # P4 = L4 * M4
            # P5 = L5 * M5
            # P6 = L6 * M6
            # P7 = L7 * M7
            
            # M1 = x+w, M2 = x, M3 = y-w, M4 = z-x, M5 = w, M6 = x+w, M7 = y+z
            
            # We need to assign columns in U, V, W for each product.
            # Current column index
            col = w_col_offset + block_idx * 7
            
            # P1
            U[0 * m + a_row_offset, col] = 1
            U[1 * m + a_row_offset + 1, col] = 1
            V[x_idx, col] = 1
            V[w_idx, col] = 1
            # W: C11 += P1, C22 += P1
            # C indices: (i, k). i in 0..1, k in 0..5.
            # C11 is (0, col_start). C22 is (1, col_start+1).
            # W index is k*n + i.
            W[col_start * n + 0, col] = 1
            W[(col_start + 1) * n + 1, col] = 1
            
            # P2
            col += 1
            U[1 * m + a_row_offset, col] = 1
            U[1 * m + a_row_offset + 1, col] = 1
            V[x_idx, col] = 1
            # C21 += P2
            W[col_start * n + 1, col] = 1
            
            # P3
            col += 1
            U[0 * m + a_row_offset, col] = 1
            V[y_idx, col] = 1
            V[w_idx, col] = -1
            # C12 += P3, C22 += P3
            W[(col_start + 1) * n + 0, col] = 1
            W[(col_start + 1) * n + 1, col] = 1
            
            # P4
            col += 1
            U[1 * m + a_row_offset + 1, col] = 1
            V[z_idx, col] = 1
            V[x_idx, col] = -1
            # C11 += P4, C21 += P4
            W[col_start * n + 0, col] = 1
            W[col_start * n + 1, col] = 1
            
            # P5
            col += 1
            U[0 * m + a_row_offset, col] = 1
            U[0 * m + a_row_offset + 1, col] = 1
            V[w_idx, col] = 1
            # C11 -= P5, C12 += P5
            W[col_start * n + 0, col] = -1
            W[(col_start + 1) * n + 0, col] = 1
            
            # P6
            col += 1
            U[1 * m + a_row_offset, col] = 1
            U[1 * m + a_row_offset + 1, col] = -1
            V[x_idx, col] = 1
            V[w_idx, col] = 1
            # C22 += P6
            W[(col_start + 1) * n + 1, col] = 1
            
            # P7
            col += 1
            U[0 * m + a_row_offset + 1, col] = 1
            U[1 * m + a_row_offset + 1, col] = -1
            V[y_idx, col] = 1
            V[z_idx, col] = 1
            # C11 += P7
            W[col_start * n + 0, col] = 1
            
    # Block 1: A1 (cols 0,1), B1 (rows 0,1)
    add_strassen_block(0, 0, 0)
    
    # Block 2: A2 (cols 2,3), B2 (rows 2,3)
    add_strassen_block(2, 2, 21)
    
    return {"U": U, "V": V, "W": W}
