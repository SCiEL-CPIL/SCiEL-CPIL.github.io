"""Matrix multiplication 6x6x6 tensor decomposition via Strassen (2x2 blocks of 3x3)."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 decomposition: 2x2 grid of 3x3 blocks with Strassen at block level
    and schoolbook (rank 27) for each 3x3x3 block product.
    Total rank = 7 * 27 = 189.
    """
    n, m, p = DIMENSIONS  # all 6
    block = 3  # 3x3 blocks, 2x2 grid
    rank = 7 * 27  # 189
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen products for 2x2 block matrices:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Strassen A-side coefficients: dict mapping (block_row, block_col) -> coeff
    strassen_A = [
        {(0, 0): 1, (1, 1): 1},          # M1: A11 + A22
        {(1, 0): 1, (1, 1): 1},          # M2: A21 + A22
        {(0, 0): 1},                     # M3: A11
        {(1, 1): 1},                     # M4: A22
        {(0, 0): 1, (0, 1): 1},          # M5: A11 + A12
        {(1, 0): 1, (0, 0): -1},         # M6: A21 - A11
        {(0, 1): 1, (1, 1): -1},         # M7: A12 - A22
    ]
    
    # Strassen B-side coefficients
    strassen_B = [
        {(0, 0): 1, (1, 1): 1},          # M1: B11 + B22
        {(0, 0): 1},                     # M2: B11
        {(0, 1): 1, (1, 1): -1},         # M3: B12 - B22
        {(1, 0): 1, (0, 0): -1},         # M4: B21 - B11
        {(1, 1): 1},                     # M5: B22
        {(0, 0): 1, (0, 1): 1},          # M6: B11 + B12
        {(1, 0): 1, (1, 1): 1},          # M7: B21 + B22
    ]
    
    # C output block contributions from Strassen:
    # C11 = M1 + M4 - M5 + M7  -> C block (0,0)
    # C12 = M3 + M5            -> C block (0,1)
    # C21 = M2 + M4            -> C block (1,0)
    # C22 = M1 - M2 + M3 + M6  -> C block (1,1)
    
    # For each Strassen product si, define which C blocks it contributes to with what coefficient
    strassen_C = [
        # si=0 (M1): C11 += 1, C22 += 1
        [(0, 0, 1), (1, 1, 1)],
        # si=1 (M2): C21 += 1, C22 -= 1
        [(1, 0, 1), (1, 1, -1)],
        # si=2 (M3): C12 += 1, C22 += 1
        [(0, 1, 1), (1, 1, 1)],
        # si=3 (M4): C11 += 1, C21 += 1
        [(0, 0, 1), (1, 0, 1)],
        # si=4 (M5): C11 -= 1, C12 += 1
        [(0, 0, -1), (0, 1, 1)],
        # si=5 (M6): C22 += 1
        [(1, 1, 1)],
        # si=6 (M7): C11 += 1
        [(0, 0, 1)],
    ]
    
    for si in range(7):
        A_coeff = strassen_A[si]
        B_coeff = strassen_B[si]
        C_contrib = strassen_C[si]
        
        for a in range(block):
            for b in range(block):
                for c in range(block):
                    col = si * 27 + a * 9 + b * 3 + c
                    
                    # U: picks up A[i][j] where i = br*3+a, j = bc*3+b
                    for (br, bc), coeff in A_coeff.items():
                        row_U = (br * block + a) * m + (bc * block + b)
                        U[row_U, col] += coeff
                    
                    # V: picks up B[j][k] where j = br*3+b, k = bc*3+c
                    for (br, bc), coeff in B_coeff.items():
                        row_V = (br * block + b) * p + (bc * block + c)
                        V[row_V, col] += coeff
                    
                    # W: picks up C[i][k] where i = brC*3+a, k = bcC*3+c
                    # Convention: W[k*n + i, r]
                    for (brC, bcC, coeff) in C_contrib:
                        row_W = (bcC * block + c) * n + (brC * block + a)
                        W[row_W, col] += coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
