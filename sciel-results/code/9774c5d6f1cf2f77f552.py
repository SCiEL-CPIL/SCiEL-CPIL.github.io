"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We treat the 3x8 matrix B as two 3x4 blocks.
    # For each 3x4 block, we split A (2x3) into A22 (2x2) and A21 (2x1).
    # We split B (3x4) into B24 (2x4) and B14 (1x4).
    # C = A22 * B24 + A21 * B14.
    # B24 is 2x4, split into two 2x2 blocks.
    # Rank for A22 * B24 is 7 + 7 = 14 (Strassen for each 2x2).
    # Rank for A21 * B14 is 2 * 4 = 8 (Outer product).
    # Total per 3x4 block: 14 + 8 = 22.
    # Total for two blocks: 22 + 22 = 44.
    
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen decomposition for 2x2 * 2x2
    # A indices: (0,0), (0,1), (1,0), (1,1) in local 2x2 grid
    # B indices: (0,0), (0,1), (1,0), (1,1) in local 2x2 grid
    # C indices: (0,0), (0,1), (1,0), (1,1) in local 2x2 grid
    def add_strassen_2x2(a_base, b_base, c_base, col_start):
        # a_base, b_base, c_base are (row, col) offsets in the full matrices
        # U index: a_row * m + a_col
        # V index: b_row * p + b_col
        # W index: c_col * n + c_row  (Note: W[k*n+i, r] where k is col of C, i is row of C)
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[a_base[0]*m + a_base[1]][col_start] = 1
        U[(a_base[0]+1)*m + (a_base[1]+1)][col_start] = 1
        V[b_base[0]*p + b_base[1]][col_start] = 1
        V[(b_base[0]+1)*p + (b_base[1]+1)][col_start] = 1
        W[(c_base[1]+0)*n + (c_base[0]+0)][col_start] = 1 # C00 += M1
        W[(c_base[1]+1)*n + (c_base[0]+1)][col_start] = 1 # C11 += M1
        col_start += 1
        
        # M2 = (A10 + A11) * B00
        U[(a_base[0]+1)*m + (a_base[1]+0)][col_start] = 1
        U[(a_base[0]+1)*m + (a_base[1]+1)][col_start] = 1
        V[b_base[0]*p + b_base[1]][col_start] = 1
        W[(c_base[1]+0)*n + (c_base[0]+1)][col_start] = 1 # C10 += M2
        W[(c_base[1]+1)*n + (c_base[0]+1)][col_start] = -1 # C11 -= M2
        col_start += 1
        
        # M3 = A00 * (B01 - B11)
        U[a_base[0]*m + a_base[1]][col_start] = 1
        V[b_base[0]*p + (b_base[1]+1)][col_start] = 1
        V[(b_base[0]+1)*p + (b_base[1]+1)][col_start] = -1
        W[(c_base[1]+1)*n + (c_base[0]+0)][col_start] = 1 # C01 += M3
        W[(c_base[1]+1)*n + (c_base[0]+1)][col_start] = 1 # C11 += M3
        col_start += 1
        
        # M4 = A11 * (B10 - B00)
        U[(a_base[0]+1)*m + (a_base[1]+1)][col_start] = 1
        V[(b_base[0]+1)*p + (b_base[1]+0)][col_start] = 1
        V[b_base[0]*p + b_base[1]][col_start] = -1
        W[(c_base[1]+0)*n + (c_base[0]+0)][col_start] = 1 # C00 += M4
        W[(c_base[1]+0)*n + (c_base[0]+1)][col_start] = 1 # C10 += M4
        col_start += 1
        
        # M5 = (A00 + A01) * B11
        U[a_base[0]*m + a_base[1]][col_start] = 1
        U[a_base[0]*m + (a_base[1]+1)][col_start] = 1
        V[(b_base[0]+1)*p + (b_base[1]+1)][col_start] = 1
        W[(c_base[1]+0)*n + (c_base[0]+0)][col_start] = -1 # C00 -= M5
        W[(c_base[1]+1)*n + (c_base[0]+0)][col_start] = 1 # C01 += M5
        col_start += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[(a_base[0]+1)*m + (a_base[1]+0)][col_start] = 1
        U[a_base[0]*m + a_base[1]][col_start] = -1
        V[b_base[0]*p + b_base[1]][col_start] = 1
        V[b_base[0]*p + (b_base[1]+1)][col_start] = 1
        W[(c_base[1]+1)*n + (c_base[0]+1)][col_start] = 1 # C11 += M6
        col_start += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[a_base[0]*m + (a_base[1]+1)][col_start] = 1
        U[(a_base[0]+1)*m + (a_base[1]+1)][col_start] = -1
        V[(b_base[0]+1)*p + (b_base[1]+0)][col_start] = 1
        V[(b_base[0]+1)*p + (b_base[1]+1)][col_start] = 1
        W[(c_base[1]+0)*n + (c_base[0]+0)][col_start] = 1 # C00 += M7
        col_start += 1
        
        return col_start

    # Process two 3x4 blocks
    for block_idx in range(2):
        # B columns for this block: 4*block_idx to 4*block_idx + 3
        b_col_start = 4 * block_idx
        # C columns for this block: 4*block_idx to 4*block_idx + 3
        c_col_start = 4 * block_idx
        
        # --- Part 1: A22 (2x2) * B24 (2x4) ---
        # B24 is rows 0,1 of B, cols b_col_start to b_col_start+3
        # Split B24 into two 2x2 blocks:
        # Block 1: cols b_col_start, b_col_start+1
        # Block 2: cols b_col_start+2, b_col_start+3
        
        # A22 is rows 0,1 of A, cols 0,1
        a_base = (0, 0)
        
        # Strassen for first 2x2 block of B24
        b_base1 = (0, b_col_start)
        c_base1 = (0, c_col_start)
        col = add_strassen_2x2(a_base, b_base1, c_base1, col)
        
        # Strassen for second 2x2 block of B24
        b_base2 = (0, b_col_start + 2)
        c_base2 = (0, c_col_start + 2)
        col = add_strassen_2x2(a_base, b_base2, c_base2, col)
        
        # --- Part 2: A21 (2x1) * B14 (1x4) ---
        # A21 is rows 0,1 of A, col 2
        # B14 is row 2 of B, cols b_col_start to b_col_start+3
        # C14 is rows 0,1 of C, cols c_col_start to c_col_start+3
        
        # A21 has entries A[0,2], A[1,2]
        # B14 has entries B[2, c] for c in b_col_start..b_col_start+3
        # C14 has entries C[i, c] = A[i,2] * B[2, c]
        
        for i in range(2): # Row of A21
            for j in range(4): # Col of B14
                U[i * m + 2][col] = 1 # A[i, 2]
                V[2 * p + (b_col_start + j)][col] = 1 # B[2, b_col_start+j]
                W[(c_col_start + j) * n + i][col] = 1 # C[i, c_col_start+j]
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
