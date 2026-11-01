"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank calculated from block decomposition
    # Strassen 2x2x2 (rank 7) x 3 blocks (cols 0-1, 2-3, 4-5)
    # Schoolbook 2x2x1 (rank 4) x 1 block (col 6)
    # Schoolbook 2x1x2 (rank 4) x 2 blocks
    # Schoolbook 2x1x3 (rank 6) x 1 block
    # Schoolbook 1x2x2 (rank 4) x 2 blocks
    # Schoolbook 1x2x3 (rank 6) x 1 block
    # Schoolbook 1x1x2 (rank 2) x 2 blocks
    # Schoolbook 1x1x3 (rank 3) x 1 block
    
    rank = 0
    # Calculate exact rank
    rank += 7 * 3  # Strassen for A22*B22 (cols 0-5)
    rank += 4      # Schoolbook for A22*B23 (col 6)
    rank += 4 * 2  # Schoolbook for A21*B12 (cols 0-3)
    rank += 6      # Schoolbook for A21*B13 (cols 4-6)
    rank += 4 * 2  # Schoolbook for A12*B22 (cols 0-3)
    rank += 6      # Schoolbook for A12*B23 (cols 4-6)
    rank += 2 * 2  # Schoolbook for A11*B12 (cols 0-3)
    rank += 3      # Schoolbook for A11*B13 (cols 4-6)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    def add_strassen_block(i_off, j_off, k_off, start_col):
        nonlocal col_idx
        # Strassen 2x2x2
        # Local indices 0,1
        # M1 = A00(B10-B11)
        U[i_off*3 + 0*3 + 0][start_col] += 1
        V[j_off*7 + 1*7 + 0][start_col] += 1
        V[j_off*7 + 1*7 + 1][start_col] += -1
        W[(k_off+0)*3 + i_off][start_col] += 1 # C00
        W[(k_off+1)*3 + i_off][start_col] += 1 # C01
        col_idx += 1
        
        # M2 = (A00+A10)B11
        U[i_off*3 + 0*3 + 0][start_col] += 1
        U[i_off*3 + 1*3 + 0][start_col] += 1
        V[j_off*7 + 1*7 + 1][start_col] += 1
        W[(k_off+0)*3 + i_off][start_col] += -1 # C00
        W[(k_off+1)*3 + i_off][start_col] += 1 # C01
        col_idx += 1
        
        # M3 = (A01+A11)B00
        U[i_off*3 + 0*3 + 1][start_col] += 1
        U[i_off*3 + 1*3 + 1][start_col] += 1
        V[j_off*7 + 0*7 + 0][start_col] += 1
        W[(k_off+0)*3 + (i_off+1)][start_col] += 1 # C10
        W[(k_off+1)*3 + (i_off+1)][start_col] += 1 # C11
        col_idx += 1
        
        # M4 = A11(B01-B00)
        U[i_off*3 + 1*3 + 1][start_col] += 1
        V[j_off*7 + 0*7 + 1][start_col] += 1
        V[j_off*7 + 0*7 + 0][start_col] += -1
        W[(k_off+0)*3 + i_off][start_col] += 1 # C00
        W[(k_off+1)*3 + (i_off+1)][start_col] += 1 # C10
        col_idx += 1
        
        # M5 = (A00+A11)(B00+B11)
        U[i_off*3 + 0*3 + 0][start_col] += 1
        U[i_off*3 + 1*3 + 1][start_col] += 1
        V[j_off*7 + 0*7 + 0][start_col] += 1
        V[j_off*7 + 1*7 + 1][start_col] += 1
        W[(k_off+0)*3 + i_off][start_col] += 1 # C00
        W[(k_off+1)*3 + (i_off+1)][start_col] += 1 # C11
        col_idx += 1
        
        # M6 = (A01-A00)(B00+B10)
        U[i_off*3 + 0*3 + 1][start_col] += 1
        U[i_off*3 + 0*3 + 0][start_col] += -1
        V[j_off*7 + 0*7 + 0][start_col] += 1
        V[j_off*7 + 1*7 + 0][start_col] += 1
        W[(k_off+0)*3 + i_off][start_col] += 1 # C00
        col_idx += 1
        
        # M7 = (A10-A11)(B01+B11)
        U[i_off*3 + 1*3 + 0][start_col] += 1
        U[i_off*3 + 1*3 + 1][start_col] += -1
        V[j_off*7 + 0*7 + 1][start_col] += 1
        V[j_off*7 + 1*7 + 1][start_col] += 1
        W[(k_off+1)*3 + (i_off+1)][start_col] += 1 # C11
        col_idx += 1

    def add_schoolbook_block(i_off, j_off, k_off, n_b, m_b, p_b, start_col):
        nonlocal col_idx
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u_idx = (i_off + i) * 3 + (j_off + j)
                    v_idx = (j_off + j) * 7 + (k_off + k)
                    w_idx = (k_off + k) * 3 + (i_off + i)
                    U[u_idx][start_col] += 1
                    V[v_idx][start_col] += 1
                    W[w_idx][start_col] += 1
                    col_idx += 1

    # Block 1: A22 (0:2, 0:2) * B22 (0:2, 0:2) -> C22 (0:2, 0:2)
    # i_off=0, j_off=0, k_off=0
    add_strassen_block(0, 0, 0, col_idx)
    
    # Block 2: A22 (0:2, 0:2) * B22 (0:2, 2:4) -> C22 (0:2, 2:4)
    # i_off=0, j_off=0, k_off=2
    add_strassen_block(0, 0, 2, col_idx)
    
    # Block 3: A22 (0:2, 0:2) * B22 (0:2, 4:6) -> C22 (0:2, 4:6)
    # i_off=0, j_off=0, k_off=4
    add_strassen_block(0, 0, 4, col_idx)
    
    # Block 4: A22 (0:2, 0:2) * B21 (0:2, 6:7) -> C21 (0:2, 6:7)
    # i_off=0, j_off=0, k_off=6, size 2x2x1
    add_schoolbook_block(0, 0, 6, 2, 2, 1, col_idx)
    
    # Block 5: A21 (0:2, 2:3) * B12 (2:3, 0:2) -> C22 (0:2, 0:2)
    # i_off=0, j_off=2, k_off=0, size 2x1x2
    add_schoolbook_block(0, 2, 0, 2, 1, 2, col_idx)
    
    # Block 6: A21 (0:2, 2:3) * B12 (2:3, 2:4) -> C22 (0:2, 2:4)
    # i_off=0, j_off=2, k_off=2, size 2x1x2
    add_schoolbook_block(0, 2, 2, 2, 1, 2, col_idx)
    
    # Block 7: A21 (0:2, 2:3) * B13 (2:3, 4:7) -> C23 (0:2, 4:7)
    # i_off=0, j_off=2, k_off=4, size 2x1x3
    add_schoolbook_block(0, 2, 4, 2, 1, 3, col_idx)
    
    # Block 8: A12 (2:3, 0:2) * B22 (0:2, 0:2) -> C12 (2:3, 0:2)
    # i_off=2, j_off=0, k_off=0, size 1x2x2
    add_schoolbook_block(2, 0, 0, 1, 2, 2, col_idx)
    
    # Block 9: A12 (2:3, 0:2) * B22 (0:2, 2:4) -> C12 (2:3, 2:4)
    # i_off=2, j_off=0, k_off=2, size 1x2x2
    add_schoolbook_block(2, 0, 2, 1, 2, 2, col_idx)
    
    # Block 10: A12 (2:3, 0:2) * B23 (0:2, 4:7) -> C13 (2:3, 4:7)
    # i_off=2, j_off=0, k_off=4, size 1x2x3
    add_schoolbook_block(2, 0, 4, 1, 2, 3, col_idx)
    
    # Block 11: A11 (2:3, 2:3) * B12 (2:3, 0:2) -> C12 (2:3, 0:2)
    # i_off=2, j_off=2, k_off=0, size 1x1x2
    add_schoolbook_block(2, 2, 0, 1, 1, 2, col_idx)
    
    # Block 12: A11 (2:3, 2:3) * B12 (2:3, 2:4) -> C12 (2:3, 2:4)
    # i_off=2, j_off=2, k_off=2, size 1x1x2
    add_schoolbook_block(2, 2, 2, 1, 1, 2, col_idx)
    
    # Block 13: A11 (2:3, 2:3) * B13 (2:3, 4:7) -> C13 (2:3, 4:7)
    # i_off=2, j_off=2, k_off=4, size 1x1x3
    add_schoolbook_block(2, 2, 4, 1, 1, 3, col_idx)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
