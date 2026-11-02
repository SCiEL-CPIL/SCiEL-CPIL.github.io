"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank: n*m*p = 48
    rank_school = n * m * p
    
    # Strassen for 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B10)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B11 + B10)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For 2x3 * 3x8:
    # Split B into two 3x4 blocks: B1 (cols 0-3), B2 (cols 4-7)
    # For each 2x3 * 3x4 block:
    # Split A into A1 (2x2, cols 0-1) and A2 (2x1, col 2)
    # Split B_block into B_block1 (3x2, cols 0-1), B_block2 (3x2, cols 2-3)
    # and B_row2 (1x4, row 2)
    # 
    # Product = A1 * [B_block1, B_block2] + A2 * B_row2
    # = [A1*B_block1, A1*B_block2] + [A2*B_row2, A2*B_row2]  (but A2 is 2x1, B_row2 is 1x4)
    # 
    # So for each 2x3 * 3x4 block:
    # A1 (2x2) * B_block1 (3x2) - but B_block1 is 3x2, not 2x2.
    # Actually B_block is 3x4. Let's split rows of B_block:
    # B_block = [B_top (2x4), B_bottom (1x4)] where B_top has rows 0,1 and B_bottom has row 2.
    # 
    # Product = A1 * B_top + A2 * B_bottom
    # where A1 is 2x2, B_top is 2x4, A2 is 2x1, B_bottom is 1x4.
    # 
    # A1 * B_top: 2x2 * 2x4 -> use Strassen on the 2x2 * 2x2 blocks of B_top
    # B_top = [B_top_1 (2x2), B_top_2 (2x2)]
    # A1 * B_top_1: rank 7
    # A1 * B_top_2: rank 7
    # A2 * B_bottom: 2x1 * 1x4 = 8 scalar multiplications, rank 8
    # Total per block: 7 + 7 + 8 = 22
    # Two blocks: 44 total.

    # Let's build the rank-44 decomposition.
    # 
    # For each block b in {0, 1} (b=0: cols 0-3, b=1: cols 4-7):
    #   A1 is rows i in {0,1}, cols j in {0,1}
    #   A2 is rows i in {0,1}, col j=2
    #   B_top rows j in {0,1}, cols k in {b*4, b*4+3}
    #   B_bottom row j=2, cols k in {b*4, b*4+3}
    #
    # For A1 * B_top_1 (2x2 * 2x2), Strassen terms:
    # B_top_1 has cols k in {b*4, b*4+1}
    # 
    # For A1 * B_top_2 (2x2 * 2x2), Strassen terms:
    # B_top_2 has cols k in {b*4+2, b*4+3}
    #
    # For A2 * B_bottom:
    # A2 has col j=2, B_bottom has row j=2, cols k in {b*4, b*4+3}

    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 44

    col = 0
    for b in range(2):
        k_base = b * 4
        
        # Strassen for A1 * B_top_1 (cols k_base, k_base+1)
        # M1 = (A00 + A11) * (B00 + B11)
        # B_top_1: row j=0 -> col k_base, row j=1 -> col k_base+1
        # So B00 = B[0, k_base], B11 = B[1, k_base+1]
        # B01 = B[0, k_base+1], B10 = B[1, k_base]
        
        # M1
        U[0*m + 0][col] += 1  # A00
        U[1*m + 1][col] += 1  # A11
        V[0*p + k_base][col] += 1  # B[0, k_base]
        V[1*p + (k_base+1)][col] += 1  # B[1, k_base+1]
        W[k_base*n + 0][col] += 1  # C[0, k_base]
        W[(k_base+1)*n + 1][col] += 1  # C[1, k_base+1]
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[1*m + 0][col] += 1  # A10
        U[1*m + 1][col] += 1  # A11
        V[0*p + k_base][col] += 1  # B[0, k_base]
        W[k_base*n + 1][col] += 1  # C[1, k_base]
        col += 1
        
        # M3 = A00 * (B10 - B11)
        U[0*m + 0][col] += 1  # A00
        V[1*p + k_base][col] += 1  # B[1, k_base]
        V[1*p + (k_base+1)][col] += -1  # -B[1, k_base+1]
        W[k_base*n + 0][col] += 1  # C[0, k_base]
        W[(k_base+1)*n + 0][col] += -1  # -C[0, k_base+1]
        col += 1
        
        # M4 = A11 * (B01 - B10)
        U[1*m + 1][col] += 1  # A11
        V[0*p + (k_base+1)][col] += 1  # B[0, k_base+1]
        V[1*p + k_base][col] += -1  # -B[1, k_base]
        W[(k_base+1)*n + 0][col] += 1  # C[0, k_base+1]
        W[k_base*n + 1][col] += -1  # -C[1, k_base]
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[0*m + 0][col] += 1  # A00
        U[0*m + 1][col] += 1  # A01
        V[1*p + (k_base+1)][col] += 1  # B[1, k_base+1]
        W[(k_base+1)*n + 0][col] += -1  # -C[0, k_base+1]
        W[(k_base+1)*n + 1][col] += 1  # C[1, k_base+1]
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B10)
        U[1*m + 0][col] += 1  # A10
        U[0*m + 0][col] += -1  # -A00
        V[0*p + k_base][col] += 1  # B[0, k_base]
        V[1*p + k_base][col] += 1  # B[1, k_base]
        W[k_base*n + 1][col] += 1  # C[1, k_base]
        col += 1
        
        # M7 = (A01 - A11) * (B11 + B10)
        U[0*m + 1][col] += 1  # A01
        U[1*m + 1][col] += -1  # -A11
        V[1*p + (k_base+1)][col] += 1  # B[1, k_base+1]
        V[1*p + k_base][col] += 1  # B[1, k_base]
        W[(k_base+1)*n + 0][col] += 1  # C[0, k_base+1]
        W[k_base*n + 1][col] += -1  # -C[1, k_base]
        col += 1

        # Strassen for A1 * B_top_2 (cols k_base+2, k_base+3)
        # M1 = (A00 + A11) * (B00 + B11)
        # B_top_2: row j=0 -> col k_base+2, row j=1 -> col k_base+3
        # B00 = B[0, k_base+2], B11 = B[1, k_base+3]
        # B01 = B[0, k_base+3], B10 = B[1, k_base+2]
        
        # M1
        U[0*m + 0][col] += 1  # A00
        U[1*m + 1][col] += 1  # A11
        V[0*p + (k_base+2)][col] += 1  # B[0, k_base+2]
        V[1*p + (k_base+3)][col] += 1  # B[1, k_base+3]
        W[(k_base+2)*n + 0][col] += 1  # C[0, k_base+2]
        W[(k_base+3)*n + 1][col] += 1  # C[1, k_base+3]
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[1*m + 0][col] += 1  # A10
        U[1*m + 1][col] += 1  # A11
        V[0*p + (k_base+2)][col] += 1  # B[0, k_base+2]
        W[(k_base+2)*n + 1][col] += 1  # C[1, k_base+2]
        col += 1
        
        # M3 = A00 * (B10 - B11)
        U[0*m + 0][col] += 1  # A00
        V[1*p + (k_base+2)][col] += 1  # B[1, k_base+2]
        V[1*p + (k_base+3)][col] += -1  # -B[1, k_base+3]
        W[(k_base+2)*n + 0][col] += 1  # C[0, k_base+2]
        W[(k_base+3)*n + 0][col] += -1  # -C[0, k_base+3]
        col += 1
        
        # M4 = A11 * (B01 - B10)
        U[1*m + 1][col] += 1  # A11
        V[0*p + (k_base+3)][col] += 1  # B[0, k_base+3]
        V[1*p + (k_base+2)][col] += -1  # -B[1, k_base+2]
        W[(k_base+3)*n + 0][col] += 1  # C[0, k_base+3]
        W[(k_base+2)*n + 1][col] += -1  # -C[1, k_base+2]
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[0*m + 0][col] += 1  # A00
        U[0*m + 1][col] += 1  # A01
        V[1*p + (k_base+3)][col] += 1  # B[1, k_base+3]
        W[(k_base+3)*n + 0][col] += -1  # -C[0, k_base+3]
        W[(k_base+3)*n + 1][col] += 1  # C[1, k_base+3]
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B10)
        U[1*m + 0][col] += 1  # A10
        U[0*m + 0][col] += -1  # -A00
        V[0*p + (k_base+2)][col] += 1  # B[0, k_base+2]
        V[1*p + (k_base+2)][col] += 1  # B[1, k_base+2]
        W[(k_base+2)*n + 1][col] += 1  # C[1, k_base+2]
        col += 1
        
        # M7 = (A01 - A11) * (B11 + B10)
        U[0*m + 1][col] += 1  # A01
        U[1*m + 1][col] += -1  # -A11
        V[1*p + (k_base+3)][col] += 1  # B[1, k_base+3]
        V[1*p + (k_base+2)][col] += 1  # B[1, k_base+2]
        W[(k_base+3)*n + 0][col] += 1  # C[0, k_base+3]
        W[(k_base+2)*n + 1][col] += -1  # -C[1, k_base+2]
        col += 1

        # A2 * B_bottom: 2x1 * 1x4
        # A2: row i in {0,1}, col j=2
        # B_bottom: row j=2, col k in {k_base, k_base+1, k_base+2, k_base+3}
        for k_offset in range(4):
            k = k_base + k_offset
            U[0*m + 2][col] += 1  # A[0, 2]
            U[1*m + 2][col] += 1  # A[1, 2]
            V[2*p + k][col] += 1  # B[2, k]
            W[k*n + 0][col] += 1  # C[0, k]
            W[k*n + 1][col] += 1  # C[1, k]
            col += 1

    # Verify col == rank
    assert col == rank, f"col={col}, rank={rank}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
