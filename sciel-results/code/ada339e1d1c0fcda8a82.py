"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, col, u_val=1, v_val=1, w_val=1):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Part 1: 2x1 by 1x4 (Rank 8)
    # A[i, 0] * B[0, k] -> C[i, k]
    col = 0
    for i in range(2):
        for k in range(4):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            add_term(u_idx, v_idx, w_idx, col)
            col += 1
            
    # Part 2: 2x2 by 2x4 (Rank 14)
    # Split into Left (2x2 by 2x2) and Right (2x2 by 2x2)
    
    # Strassen multipliers definition for 2x2 by 2x2
    # Inputs: X (2x2), Y (2x2)
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y11)
    # M4 = X11(Y01-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # Left Block: A cols 1,2; B rows 1,2, cols 0,1
    # X = A[0:2, 1:3], Y = B[1:3, 0:2]
    # C = C[0:2, 0:2]
    
    # Right Block: A cols 1,2; B rows 1,2, cols 2,4
    # X = A[0:2, 1:3], Y = B[1:3, 2:4]
    # C = C[0:2, 2:4]
    
    def add_strassen_block(base_col, b_row_offset, b_col_offset, c_col_offset):
        # b_row_offset is 1 (global row index of B start)
        # b_col_offset is 0 for Left, 2 for Right
        # c_col_offset is 0 for Left, 2 for Right
        
        # X indices (A):
        # X00 -> A[0, 1] -> idx 1
        # X01 -> A[0, 2] -> idx 2
        # X10 -> A[1, 1] -> idx 4
        # X11 -> A[1, 2] -> idx 5
        
        # Y indices (B):
        # Y00 -> B[b_row_offset, b_col_offset]
        # Y01 -> B[b_row_offset, b_col_offset + 1]
        # Y10 -> B[b_row_offset + 1, b_col_offset]
        # Y11 -> B[b_row_offset + 1, b_col_offset + 1]
        
        # Z indices (C):
        # Z00 -> C[0, c_col_offset]
        # Z01 -> C[0, c_col_offset + 1]
        # Z10 -> C[1, c_col_offset]
        # Z11 -> C[1, c_col_offset + 1]
        
        # M1
        c = base_col
        add_term(1, 0, 0, c) # dummy, will overwrite with specific indices
        # Let's do it properly
        U[1][c] += 1; U[5][c] += 1
        V[(b_row_offset)*p + b_col_offset][c] += 1
        V[(b_row_offset+1)*p + (b_col_offset+1)][c] += 1
        W[(c_col_offset)*n + 0][c] += 1
        W[(c_col_offset+1)*n + 1][c] += 1
        
        # M2
        c = base_col + 1
        U[4][c] += 1; U[5][c] += 1
        V[(b_row_offset)*p + b_col_offset][c] += 1
        W[(c_col_offset)*n + 1][c] += 1
        W[(c_col_offset+1)*n + 1][c] += -1
        
        # M3
        c = base_col + 2
        U[1][c] += 1
        V[(b_row_offset+1)*p + b_col_offset][c] += 1
        V[(b_row_offset+1)*p + (b_col_offset+1)][c] += -1
        W[(c_col_offset+1)*n + 0][c] += 1
        W[(c_col_offset+1)*n + 1][c] += 1
        
        # M4
        c = base_col + 3
        U[5][c] += 1
        V[(b_row_offset)*p + (b_col_offset+1)][c] += 1
        V[(b_row_offset)*p + b_col_offset][c] += -1
        W[(c_col_offset)*n + 0][c] += 1
        W[(c_col_offset)*n + 1][c] += 1
        
        # M5
        c = base_col + 4
        U[1][c] += 1; U[2][c] += 1
        V[(b_row_offset+1)*p + (b_col_offset+1)][c] += 1
        W[(c_col_offset)*n + 0][c] += -1
        W[(c_col_offset+1)*n + 0][c] += 1
        
        # M6
        c = base_col + 5
        U[4][c] += 1; U[1][c] += -1
        V[(b_row_offset)*p + b_col_offset][c] += 1
        V[(b_row_offset)*p + (b_col_offset+1)][c] += 1
        W[(c_col_offset+1)*n + 1][c] += 1
        
        # M7
        c = base_col + 6
        U[2][c] += 1; U[5][c] += -1
        V[(b_row_offset+1)*p + b_col_offset][c] += 1
        V[(b_row_offset+1)*p + (b_col_offset+1)][c] += 1
        W[(c_col_offset)*n + 0][c] += 1

    add_strassen_block(8, 1, 0, 0) # Left
    add_strassen_block(15, 1, 2, 2) # Right

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
