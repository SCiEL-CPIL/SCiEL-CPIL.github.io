import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Rank 72 decomposition:
    # 1. Use Strassen (rank 7) for 8 block multiplications of 2x2 * 2x2 to compute first 4 cols.
    #    Total 8 * 7 = 56.
    # 2. Use Trivial (rank 16) for 4x4 * 4x1 to compute last col.
    #    Total 16.
    # Sum = 72.
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col_offset, a_r, a_c, b_r, b_c, c_r, c_c):
        """
        Adds Strassen decomposition for a 2x2 * 2x2 block.
        a_r, a_c: top-left row/col of A block
        b_r, b_c: top-left row/col of B block
        c_r, c_c: top-left row/col of C block
        """
        # Local indices 0,1
        # Global A index: (a_r + i) * m + (a_c + j)
        # Global B index: (b_r + j) * p + (b_c + k)
        # Global W index: (c_c + k) * n + (c_r + i)
        
        # Term 1: m1 = (a00+a11)(b00+b11) -> c00 + c11
        U[(a_r+0)*m + (a_c+0)][col_offset] = 1
        U[(a_r+1)*m + (a_c+1)][col_offset] = 1
        V[(b_r+0)*p + (b_c+0)][col_offset] = 1
        V[(b_r+1)*p + (b_c+1)][col_offset] = 1
        W[(c_c+0)*n + (c_r+0)][col_offset] = 1
        W[(c_c+1)*n + (c_r+1)][col_offset] = 1
        
        # Term 2: m2 = (a10+a11)b00 -> c10 - c11
        col_offset += 1
        U[(a_r+1)*m + (a_c+0)][col_offset] = 1
        U[(a_r+1)*m + (a_c+1)][col_offset] = 1
        V[(b_r+0)*p + (b_c+0)][col_offset] = 1
        W[(c_c+0)*n + (c_r+1)][col_offset] = 1
        W[(c_c+1)*n + (c_r+1)][col_offset] = -1
        
        # Term 3: m3 = a00(b10-b11) -> c01 + c11
        col_offset += 1
        U[(a_r+0)*m + (a_c+0)][col_offset] = 1
        V[(b_r+1)*p + (b_c+0)][col_offset] = 1
        V[(b_r+1)*p + (b_c+1)][col_offset] = -1
        W[(c_c+1)*n + (c_r+0)][col_offset] = 1
        W[(c_c+1)*n + (c_r+1)][col_offset] = 1
        
        # Term 4: m4 = a11(b01-b00) -> c00 + c10
        col_offset += 1
        U[(a_r+1)*m + (a_c+1)][col_offset] = 1
        V[(b_r+0)*p + (b_c+1)][col_offset] = 1
        V[(b_r+0)*p + (b_c+0)][col_offset] = -1
        W[(c_c+0)*n + (c_r+0)][col_offset] = 1
        W[(c_c+0)*n + (c_r+1)][col_offset] = 1
        
        # Term 5: m5 = (a00+a01)b11 -> -c00 + c01
        col_offset += 1
        U[(a_r+0)*m + (a_c+0)][col_offset] = 1
        U[(a_r+0)*m + (a_c+1)][col_offset] = 1
        V[(b_r+1)*p + (b_c+1)][col_offset] = 1
        W[(c_c+0)*n + (c_r+0)][col_offset] = -1
        W[(c_c+1)*n + (c_r+0)][col_offset] = 1
        
        # Term 6: m6 = (a10-a00)(b00+b01) -> c11
        col_offset += 1
        U[(a_r+1)*m + (a_c+0)][col_offset] = 1
        U[(a_r+0)*m + (a_c+0)][col_offset] = -1
        V[(b_r+0)*p + (b_c+0)][col_offset] = 1
        V[(b_r+0)*p + (b_c+1)][col_offset] = 1
        W[(c_c+1)*n + (c_r+1)][col_offset] = 1
        
        # Term 7: m7 = (a01-a11)(b10+b11) -> c00
        col_offset += 1
        U[(a_r+0)*m + (a_c+1)][col_offset] = 1
        U[(a_r+1)*m + (a_c+1)][col_offset] = -1
        V[(b_r+1)*p + (b_c+0)][col_offset] = 1
        V[(b_r+1)*p + (b_c+1)][col_offset] = 1
        W[(c_c+0)*n + (c_r+0)][col_offset] = 1
        
        return col_offset + 1

    col = 0
    
    # Blocks for 4x4 * 4x4 (cols 0-3)
    # A blocks:
    # A11: r=0, c=0
    # A12: r=0, c=2
    # A21: r=2, c=0
    # A22: r=2, c=2
    
    # B blocks (cols 0-3):
    # B11: r=0, c=0
    # B12: r=0, c=2
    # B21: r=2, c=0
    # B22: r=2, c=2
    
    # C blocks (cols 0-3):
    # C11: r=0, c=0
    # C12: r=0, c=2
    # C21: r=2, c=0
    # C22: r=2, c=2
    
    # P1 = A11 B11 -> C11
    col = add_strassen_block(col, 0, 0, 0, 0, 0, 0)
    # P2 = A12 B21 -> C11
    col = add_strassen_block(col, 0, 2, 2, 0, 0, 0)
    # P3 = A11 B12 -> C12
    col = add_strassen_block(col, 0, 0, 0, 2, 0, 2)
    # P4 = A12 B22 -> C12
    col = add_strassen_block(col, 0, 2, 2, 2, 0, 2)
    # P5 = A21 B11 -> C21
    col = add_strassen_block(col, 2, 0, 0, 0, 2, 0)
    # P6 = A22 B21 -> C21
    col = add_strassen_block(col, 2, 2, 2, 0, 2, 0)
    # P7 = A21 B12 -> C22
    col = add_strassen_block(col, 2, 0, 0, 2, 2, 2)
    # P8 = A22 B22 -> C22
    col = add_strassen_block(col, 2, 2, 2, 2, 2, 2)
    
    # Trivial for 5th column (col 4)
    # C[:, 4] = A * B[:, 4]
    for i in range(n):
        for j in range(m):
            if col >= rank: break
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
