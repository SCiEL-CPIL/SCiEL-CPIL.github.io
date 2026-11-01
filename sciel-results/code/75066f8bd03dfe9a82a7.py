import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Rank-42 exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.
    
    Uses block-partitioned Strassen: split the 4-dimensional middle into two 2x2 blocks,
    each multiplied against three 2x2 output blocks using Strassen's rank-7 algorithm.
    Total rank = 2 * 3 * 7 = 42.
    
    The W factor follows the official trace convention: W[k*n + i, r] for output C[i][k].
    """
    n, m, p = 2, 4, 6
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)  # 8 x 42
    V = np.zeros((m * p, rank), dtype=np.float64)  # 24 x 42
    W = np.zeros((p * n, rank), dtype=np.float64)  # 12 x 42
    
    # Strassen 2x2 @ 2x2 rank-7 decomposition
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a10+a11)*b00
    # P3 = a00*(b01-b11)
    # P4 = a11*(b10-b00)
    # P5 = (a00+a01)*b11
    # P6 = (a10-a00)*(b00+b01)
    # P7 = (a01-a11)*(b10+b11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # For each of the 7 products, define the U, V, W coefficient patterns
    # U pattern: {local_A_index: coefficient} where local index = row*2+col
    # V pattern: {local_B_index: coefficient} where local index = row*2+col
    # W pattern: {local_C_index: coefficient} where local index = col*2+row (trace convention)
    
    strassen_patterns = [
        # P1: (a00+a11)*(b00+b11) -> C00(+1), C11(+1)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2: (a10+a11)*b00 -> C10(+1), C11(-1)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # P3: a00*(b01-b11) -> C01(+1), C11(+1)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # P4: a11*(b10-b00) -> C00(+1), C10(+1)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # P5: (a00+a01)*b11 -> C00(-1), C01(+1)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # P6: (a10-a00)*(b00+b01) -> C11(+1)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7: (a01-a11)*(b10+b11) -> C00(+1)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    col = 0
    
    for a_block in range(2):       # left (cols 0-1) or right (cols 2-3) of A
        for b_block in range(3):   # three 2x2 blocks of B (cols 0-1, 2-3, 4-5)
            for prod_idx in range(7):
                u_pat, v_pat, w_pat = strassen_patterns[prod_idx]
                
                # U: A is 2x4, local 2x2 block at column offset a_block*2
                # local index li -> row = li//2, col = li%2
                # global U index = row * m + (a_block*2 + col) = row * 4 + a_block*2 + col
                for li, coeff in u_pat.items():
                    row = li // 2
                    col_local = li % 2
                    u_idx = row * m + (a_block * 2 + col_local)
                    U[u_idx, col] += coeff
                
                # V: B is 4x6, local 2x2 block at row offset (a_block*2), col offset b_block*2
                # Wait, we need to be careful. The middle dimension m=4 is split.
                # A is 2x4, B is 4x6.
                # We split A into left (cols 0-1) and right (cols 2-3).
                # For a_block=0 (left): A_left = A[:, 0:2], B_top = B[0:2, :]
                # For a_block=1 (right): A_right = A[:, 2:4], B_bottom = B[2:4, :]
                # B is split into three 2x2 column blocks.
                
                # V local index li -> row = li//2, col = li%2
                # global V index = (a_block*2 + row) * p + (b_block*2 + col)
                for li, coeff in v_pat.items():
                    row = li // 2
                    col_local = li % 2
                    v_idx = (a_block * 2 + row) * p + (b_block * 2 + col_local)
                    V[v_idx, col] += coeff
                
                # W: Output C is 2x6, local 2x2 block at col offset b_block*2
                # local index li in W pattern uses trace convention: li = col*2 + row
                # So col_local = li // 2, row = li % 2
                # global W index = (b_block*2 + col_local) * n + row
                for li, coeff in w_pat.items():
                    col_local = li // 2
                    row = li % 2
                    w_idx = (b_block * 2 + col_local) * n + row
                    W[w_idx, col] += coeff
                
                col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
