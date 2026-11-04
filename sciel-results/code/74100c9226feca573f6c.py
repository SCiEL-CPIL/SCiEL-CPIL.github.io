"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-189 exact tensor decomposition for 6x6 matrix multiplication.
    
    Strategy: Use Strassen's algorithm (rank 7) for the 2x2 block structure,
    then schoolbook (rank 27) for each 3x3 leaf block.
    Total rank: 7 * 27 = 189 (vs schoolbook 216).
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # We decompose 6x6 into 2x2 blocks of 3x3 matrices.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Strassen for 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a product of two 3x3 matrices (sums/differences of blocks).
    # For each Mi, we need to compute a 3x3 * 3x3 product using rank-27 schoolbook.
    
    # The overall structure:
    # - 7 Strassen multipliers
    # - Each multiplier involves linear combinations of 3x3 sub-blocks
    # - Each 3x3 * 3x3 product uses rank 27
    # - Total rank = 7 * 27 = 189
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Block structure: 2x2 blocks of 3x3
    # Block (bi, bj) of A occupies rows [bi*3 : (bi+1)*3] and cols [bj*3 : (bj+1)*3]
    # Similarly for B and C.
    
    # For each Strassen multiplier s (0..6) and each schoolbook term t (0..26):
    # We need to identify:
    # - Which A-blocks and B-blocks contribute, with what signs
    # - The schoolbook term (i,j,k) within the 3x3 product
    
    # Strassen multiplier definitions:
    # s=0: M1 = (A11 + A22) * (B11 + B22)
    # s=1: M2 = (A21 + A22) * B11
    # s=2: M3 = A11 * (B12 - B22)
    # s=3: M4 = A22 * (B21 - B11)
    # s=4: M5 = (A11 + A12) * B22
    # s=5: M6 = (A21 - A11) * (B11 + B12)
    # s=6: M7 = (A12 - A22) * (B21 + B22)
    
    # For each s, define the A-side and B-side linear combinations
    # A-side: list of (block_row, block_col, sign)
    # B-side: list of (block_row, block_col, sign)
    
    strassen_defs = [
        # s=0: (A11 + A22) * (B11 + B22)
        {'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)]},
        # s=1: (A21 + A22) * B11
        {'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)]},
        # s=2: A11 * (B12 - B22)
        {'A': [(0,0,1)], 'B': [(0,1,1), (1,1,-1)]},
        # s=3: A22 * (B21 - B11)
        {'A': [(1,1,1)], 'B': [(1,0,1), (0,0,-1)]},
        # s=4: (A11 + A12) * B22
        {'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)]},
        # s=5: (A21 - A11) * (B11 + B12)
        {'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)]},
        # s=6: (A12 - A22) * (B21 + B22)
        {'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)]},
    ]
    
    # C-block contributions from each Strassen multiplier:
    # C11 = M1 + M4 - M5 + M7  => C-block (0,0)
    # C12 = M3 + M5            => C-block (0,1)
    # C21 = M2 + M4            => C-block (1,0)
    # C22 = M1 - M2 + M3 + M6  => C-block (1,1)
    
    # For each C-block, list of (s, sign)
    c_block_contribs = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],   # M1 + M4 - M5 + M7
        (0,1): [(2,1), (4,1)],                    # M3 + M5
        (1,0): [(1,1), (3,1)],                    # M2 + M4
        (1,1): [(0,1), (1,-1), (2,1), (5,1)],     # M1 - M2 + M3 + M6
    }
    
    # Now build the decomposition
    # For each s (0..6) and each schoolbook term (i,j,k) (0..2, 0..2, 0..2):
    # The rank index is s * 27 + (i*9 + j*3 + k)
    
    for s in range(7):
        for i in range(3):  # row within 3x3 block
            for j in range(3):  # col within 3x3 block (A-side)
                for k in range(3):  # col within 3x3 block (B-side)
                    col_idx = s * 27 + (i * 9 + j * 3 + k)
                    
                    # U side: A matrix
                    # The A-side linear combination for multiplier s
                    # For schoolbook term (i,j,k), we need A[row, col] where
                    # row = i (within block), col = j (within block)
                    for (br, bc, sign) in strassen_defs[s]['A']:
                        # Block (br, bc) of A: rows br*3..br*3+2, cols bc*3..bc*3+2
                        # Element (i, j) within this block is at global (br*3+i, bc*3+j)
                        U[br*3 + i][bc*3 + j] += sign
                    
                    # V side: B matrix
                    for (br, bc, sign) in strassen_defs[s]['B']:
                        # Block (br, bc) of B: rows br*3..br*3+2, cols bc*3..bc*3+2
                        # Element (j, k) within this block is at global (br*3+j, bc*3+k)
                        V[br*3 + j][bc*3 + k] += sign
                    
                    # W side: C matrix
                    # For schoolbook term (i,j,k), the contribution goes to C[i, k]
                    # within each C-block that this multiplier contributes to
                    for (cb_r, cb_c) in c_block_contribs:
                        for (s_contrib, sign_c) in c_block_contribs[(cb_r, cb_c)]:
                            if s_contrib == s:
                                # C-block (cb_r, cb_c): element (i, k) at global (cb_r*3+i, cb_c*3+k)
                                W[k * n + i][cb_r * 3 + cb_c * 3 + k] += sign_c
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
