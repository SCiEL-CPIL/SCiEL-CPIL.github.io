import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Total rank: 49 (for 4x4x4) + 28 (for 4x4x2) = 77
    total_rank = 77
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Strassen 2x2x2 factors
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Helper to add a 2x2x2 decomposition to U, V, W
    # block_a: (row_offset, col_offset) of the 2x2 block in A
    # block_b: (row_offset, col_offset) of the 2x2 block in B
    # block_c: (row_offset, col_offset) of the 2x2 block in C
    # offset: column offset in U, V, W
    def add_222(block_a, block_b, block_c, offset):
        for di in range(2):
            for dj in range(2):
                i = block_a[0] + di
                j = block_a[1] + dj
                for t in range(7):
                    U[i * m + j][offset + t] += S_U[di * 2 + dj][t]
        for dj in range(2):
            for dk in range(2):
                j = block_b[0] + dj
                k = block_b[1] + dk
                for t in range(7):
                    V[j * p + k][offset + t] += S_V[dj * 2 + dk][t]
        for dk in range(2):
            for di in range(2):
                k = block_c[1] + dk # C is k*n+i, so k is col, i is row
                i = block_c[0] + di
                for t in range(7):
                    W[k * n + i][offset + t] += S_W[di * 2 + dk][t]

    # Part 1: 4x4 x 4x4 (Rank 49)
    # Strassen steps for 4x4
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: (0,0), (0,1), (1,0), (1,1)
    # C blocks: (0,0), (0,1), (1,0), (1,1)
    
    # Coefficients for A blocks for each of the 7 steps
    A_coeffs = [
        [(0,0,1), (1,1,1)],       # M0: A00 + A11
        [(1,0,1), (1,1,1)],       # M1: A10 + A11
        [(0,0,1)],                 # M2: A00
        [(1,1,1)],                 # M3: A11
        [(0,0,1), (0,1,1)],       # M4: A00 + A01
        [(1,0,1), (0,0,-1)],      # M5: A10 - A00
        [(0,1,1), (1,1,-1)]       # M6: A01 - A11
    ]
    
    # Coefficients for B blocks for each of the 7 steps
    B_coeffs = [
        [(0,0,1), (1,1,1)],       # M0: B00 + B11
        [(0,0,1)],                 # M1: B00
        [(1,0,1), (1,1,-1)],      # M2: B10 - B11
        [(0,1,1), (0,0,-1)],      # M3: B01 - B00
        [(1,1,1)],                 # M4: B11
        [(0,0,1), (0,1,1)],       # M5: B00 + B01
        [(1,0,1), (1,1,1)]        # M6: B10 + B11
    ]
    
    # Coefficients for C blocks for each of the 7 steps
    C_coeffs = [
        [(0,0,1), (1,1,1)],       # M0: C00 + C11
        [(1,1,-1)],               # M1: -C11
        [(1,0,1)],                # M2: C10
        [(0,0,-1), (1,1,1)],      # M3: -C00 + C11
        [(0,1,1)],                # M4: C01
        [(0,0,-1), (1,0,1)],      # M5: -C00 + C10
        [(0,0,1), (0,1,1)]        # M6: C00 + C01
    ]
    
    for s in range(7):
        offset = s * 7
        # For each Strassen step s, we have a "virtual" 2x2 multiplication
        # We need to sum the contributions from the blocks involved
        
        # Build the U, V, W factors for this step
        # We can do this by iterating over the blocks and adding S_U, S_V, S_W scaled by coefficients
        
        # U
        for (bx, by, coeff) in A_coeffs[s]:
            # Block A_bx,by is at rows 2bx, 2bx+1 and cols 2by, 2by+1
            # We add S_U scaled by coeff
            for di in range(2):
                for dj in range(2):
                    i = 2 * bx + di
                    j = 2 * by + dj
                    for t in range(7):
                        U[i * m + j][offset + t] += coeff * S_U[di * 2 + dj][t]
                        
        # V
        for (bx, by, coeff) in B_coeffs[s]:
            for dj in range(2):
                for dk in range(2):
                    j = 2 * bx + dj
                    k = 2 * by + dk
                    for t in range(7):
                        V[j * p + k][offset + t] += coeff * S_V[dj * 2 + dk][t]
                        
        # W
        for (bx, by, coeff) in C_coeffs[s]:
            # Block C_bx,by is at rows 2by, 2by+1 and cols 2bx, 2bx+1
            # Note: C is k*n+i. k is col, i is row.
            # So C_bx,by corresponds to k in 2bx..2bx+1, i in 2by..2by+1
            for dk in range(2):
                for di in range(2):
                    k = 2 * bx + dk
                    i = 2 * by + di
                    for t in range(7):
                        W[k * n + i][offset + t] += coeff * S_W[di * 2 + dk][t]

    # Part 2: 4x4 x 4x2 (Rank 28)
    # B is cols 4,5.
    # B_top is rows 0,1 of B (cols 4,5). Block (0, 4).
    # B_bot is rows 2,3 of B (cols 4,5). Block (2, 4).
    
    # Products:
    # P0: A00 * B_top -> C_top
    # P1: A01 * B_bot -> C_top
    # P2: A10 * B_top -> C_bot
    # P3: A11 * B_bot -> C_bot
    
    offset_base = 49
    
    # P0
    add_222((0,0), (0,4), (0,0), offset_base)
    # P1
    add_222((0,2), (2,4), (0,0), offset_base + 7)
    # P2
    add_222((2,0), (0,4), (2,0), offset_base + 14)
    # P3
    add_222((2,2), (2,4), (2,0), offset_base + 21)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
