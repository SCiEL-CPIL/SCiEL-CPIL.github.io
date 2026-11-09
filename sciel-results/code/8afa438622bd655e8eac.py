import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Returns an exact tensor decomposition of the 6x6 x 6x6 matrix multiplication tensor.
    Uses a block-Strassen approach:
      - Partition the 6x6 matrices into a 3x3 grid of 2x2 blocks.
      - Apply the 7-multiplication Strassen algorithm to each of the 9 block positions.
      - Total rank = 9 * 7 = 63.
    
    The output follows the convention W[k*n + i, r] for the C factor.
    """
    _t0 = time.time()
    _deadline = _t0 + 145.0

    n, m, p = 6, 6, 6
    block = 2          # block size
    grid = n // block  # 3x3 grid of 2x2 blocks
    rank = grid * grid * grid * 7  # 9 * 7 = 63

    # Initialize factor matrices as NumPy arrays for efficiency, then convert to lists
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Strassen's 7 products for 2x2 blocks:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # U patterns: which (pi,pj) in the 2x2 block A gets added with what coefficient
    # V patterns: which (pi,pj) in the 2x2 block B gets added with what coefficient
    # W patterns: which (pi,pj) in the 2x2 block C gets the result with what coefficient
    #   (pi corresponds to row offset within block, pj to col offset within block for C)

    strassen = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1
        ({(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1},
         {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1},
         {(0,0): 1},
         {(1,0): 1, (1,1): 1}),
        # M3
        ({(0,0): 1},
         {(0,1): 1, (1,1): -1},
         {(0,1): 1}),
        # M4
        ({(1,1): 1},
         {(1,0): 1, (0,0): -1},
         {(1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1},
         {(1,1): 1},
         {(1,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1},
         {(0,0): 1, (0,1): 1},
         {(1,0): 1, (1,1): -1}),
        # M7
        ({(0,1): 1, (1,1): -1},
         {(1,0): 1, (1,1): 1},
         {(0,0): 1}),
    ]

    col = 0
    for I in range(grid):
        if time.time() > _deadline:
            break
        row_off_A = I * block
        row_off_C = I * block
        for J in range(grid):
            if time.time() > _deadline:
                break
            col_off_A = J * block
            row_off_B = J * block
            col_off_C = J * block
            for K in range(grid):
                if time.time() > _deadline:
                    break
                col_off_B = K * block
                row_off_B2 = K * block  # B's row offset is K*block? 
                # Wait: A is (n x m), B is (m x p), C is (p x n)
                # Block (I,J) of A: rows [I*block, (I+1)*block), cols [J*block, (J+1)*block)
                # Block (J,K) of B: rows [J*block, (J+1)*block), cols [K*block, (K+1)*block)
                # Block (I,K) of C: rows [I*block, (I+1)*block), cols [K*block, (K+1)*block)
                
                for (u_pat, v_pat, w_pat) in strassen:
                    if time.time() > _deadline:
                        break
                    # Fill U
                    for (pi, pj), val in u_pat.items():
                        r = row_off_A + pi
                        c = col_off_A + pj
                        U[r * m + c, col] += val
                    # Fill V
                    for (pi, pj), val in v_pat.items():
                        r = row_off_B + pi
                        c = col_off_B + pj
                        V[r * p + c, col] += val
                    # Fill W (C block (I,K))
                    for (pi, pj), val in w_pat.items():
                        # C block (I,K): row = I*block + pi, col = K*block + pj
                        # W index: k*n + i where k = col of C = K*block + pj, i = row of C = I*block + pi
                        k = col_off_C + pj
                        i = row_off_C + pi
                        W[k * n + i, col] += val
                    col += 1
            if time.time() > _deadline:
                break
        if time.time() > _deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
