import time

DIMENSIONS = [2, 4, 8]

# Strassen 2x2x2 decomposition: 7 bilinear terms
# M1 = (X00+X11)(Y00-Y11)
# M2 = (X10+X11)Y00
# M3 = X00(Y10-Y11)
# M4 = X11(Y01-Y00)
# M5 = (X00+X01)Y11
# M6 = (X10-X00)(Y00+Y10)
# M7 = (X01-X11)(Y11+Y01)
#
# Z00 = M1 - M2 + M3 + M6
# Z01 = M3 + M5
# Z10 = M2 + M4
# Z11 = M1 + M4 - M5 + M7

def construct():
    deadline = time.time() + 100  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (16, rank)
    
    # We have 2x2 blocks of X (n=2 so 1 row-block, 2 col-blocks since m=4 -> 2 blocks of 2)
    # Wait: n=2, m=4, p=8
    # X is 2x4. We can partition X into 1x2 blocks of 2x2.
    # Y is 4x8. We can partition Y into 2x4 blocks of 2x2.
    # Z is 2x8. We can partition Z into 1x4 blocks of 2x2.
    #
    # So we have:
    # X: 1 row-block, 2 col-blocks -> 2 blocks of 2x2
    # Y: 2 row-blocks, 4 col-blocks -> 8 blocks of 2x2
    # Z: 1 row-block, 4 col-blocks -> 4 blocks of 2x2
    #
    # For each pair (X_col_block, Y_row_block) where X_col_block and Y_row_block must match,
    # we get a 2x2 output block Z[row_block_X, col_block_Y]
    #
    # X is indexed as X[i, j] where i in [0,2), j in [0,4)
    # Y is indexed as Y[j, k] where j in [0,4), k in [0,8)
    # Z is indexed as Z[i, k] where i in [0,2), k in [0,8)
    #
    # Partition:
    # X: rows [0:2], cols [0:4] -> split cols into [0:2], [2:4] -> 2 blocks
    # Y: rows [0:4], cols [0:8] -> split rows into [0:2], [2:4], cols into [0:2], [2:4], [4:6], [6:8] -> 8 blocks
    # Z: rows [0:2], cols [0:8] -> split cols into [0:2], [2:4], [4:6], [6:8] -> 4 blocks
    #
    # For each xb in [0,2), yb_row in [0,2), yb_col in [0,4):
    #   if xb == yb_row (since X col index must match Y row index):
    #     Z_block = Z[0:2, 2*yb_col : 2*yb_col+2]
    #     X_block = X[0:2, 2*xb : 2*xb+2]
    #     Y_block = Y[2*yb_row : 2*yb_row+2, 2*yb_col : 2*yb_col+2]
    #
    # So valid combinations: xb in [0,2), yb_col in [0,4) -> 8 Strassen decompositions
    # Each uses 7 bilinear terms -> total rank = 8 * 7 = 56
    
    rank = 8 * 7
    
    # U: shape (n*m, rank) = (8, 56)
    # U[i*m + j, c] where i in [0,n), j in [0,m)
    # V: shape (m*p, rank) = (32, 56)
    # V[j*p + k, c] where j in [0,m), k in [0,p)
    # W: shape (p*n, rank) = (16, 56)
    # W[k*n + i, c] where k in [0,p), i in [0,n)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for each of the 7 terms
    # For each term t (0-6), we need:
    # - U coefficients: mapping from local X indices (i, j') to coefficient
    # - V coefficients: mapping from local Y indices (j', k') to coefficient
    # - W coefficients: mapping from local Z indices (i, k') to coefficient
    #
    # Local X: x[i, j'] where i in [0,2), j' in [0,2)
    # Local Y: y[j', k'] where j' in [0,2), k' in [0,2)
    # Local Z: z[i, k'] where i in [0,2), k' in [0,2)
    #
    # Term definitions (1-indexed as M1..M7, we use 0-indexed t=0..6):
    # M1 (t=0): (X00+X11)(Y00-Y11)
    #   U: X00->1, X11->1
    #   V: Y00->1, Y11->-1
    #   W: Z00->1, Z11->1
    # M2 (t=1): (X10+X11)Y00
    #   U: X10->1, X11->1
    #   V: Y00->1
    #   W: Z10->1, Z11->-1
    # M3 (t=2): X00(Y10-Y11)
    #   U: X00->1
    #   V: Y10->1, Y11->-1
    #   W: Z01->1, Z11->1
    # M4 (t=3): X11(Y01-Y00)
    #   U: X11->1
    #   V: Y01->1, Y00->-1
    #   W: Z00->1, Z10->1
    # M5 (t=4): (X00+X01)Y11
    #   U: X00->1, X01->1
    #   V: Y11->1
    #   W: Z00->-1, Z01->1
    # M6 (t=5): (X10-X00)(Y00+Y10)
    #   U: X10->1, X00->-1
    #   V: Y00->1, Y10->1
    #   W: Z11->1
    # M7 (t=6): (X01-X11)(Y11+Y01)
    #   U: X01->1, X11->-1
    #   V: Y11->1, Y01->1
    #   W: Z00->1
    
    # Define the coefficient tables
    # u_coeff[t] = dict mapping (i, j') to coeff
    # v_coeff[t] = dict mapping (j', k') to coeff
    # w_coeff[t] = dict mapping (i, k') to coeff
    
    u_coeff = [
        {(0,0): 1, (1,1): 1},          # M1
        {(1,0): 1, (1,1): 1},          # M2
        {(0,0): 1},                     # M3
        {(1,1): 1},                     # M4
        {(0,0): 1, (0,1): 1},          # M5
        {(1,0): 1, (0,0): -1},         # M6
        {(0,1): 1, (1,1): -1},         # M7
    ]
    
    v_coeff = [
        {(0,0): 1, (1,1): -1},         # M1
        {(0,0): 1},                     # M2
        {(1,0): 1, (1,1): -1},         # M3
        {(0,1): 1, (0,0): -1},         # M4
        {(1,1): 1},                     # M5
        {(0,0): 1, (1,0): 1},          # M6
        {(1,1): 1, (0,1): 1},          # M7
    ]
    
    w_coeff = [
        {(0,0): 1, (1,1): 1},          # M1
        {(1,0): 1, (1,1): -1},         # M2
        {(0,1): 1, (1,1): 1},          # M3
        {(0,0): 1, (1,0): 1},          # M4
        {(0,0): -1, (0,1): 1},         # M5
        {(1,1): 1},                     # M6
        {(0,0): 1},                     # M7
    ]
    
    col = 0
    for xb in range(2):          # X column block (also Y row block)
        for yb_col in range(4):  # Y column block
            for t in range(7):   # Strassen term
                # Global indices
                # X global: i in [0,2), j = 2*xb + j'
                # Y global: j = 2*xb + j', k = 2*yb_col + k'
                # Z global: i, k = 2*yb_col + k'
                
                for (li, lj), coeff_u in u_coeff[t].items():
                    gi = li
                    gj = 2 * xb + lj
                    U[gi * m + gj][col] += coeff_u
                
                for (lj, lk), coeff_v in v_coeff[t].items():
                    gj = 2 * xb + lj
                    gk = 2 * yb_col + lk
                    V[gj * p + gk][col] += coeff_v
                
                for (li, lk), coeff_w in w_coeff[t].items():
                    gi = li
                    gk = 2 * yb_col + lk
                    W[gk * n + gi][col] += coeff_w
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
