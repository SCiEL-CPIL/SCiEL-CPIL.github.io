"""Rewritten program for 2x4 x 4x8 matrix multiplication tensor decomposition."""

import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Construct a rank-56 tensor decomposition for multiplying a 2x4 matrix by a 4x8 matrix.
    
    Uses 8 Strassen 2x2x2 blocks (each rank 7) to achieve total rank 56.
    
    Structure:
    - A is 2x4, split into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3)
    - B is 4x8, split into 8 2x2 blocks:
      B[i,j] where i in {0,1} (row block), j in {0,1,2,3} (col block of 2)
    - C = A_L * B[0,:] + A_R * B[1,:]
    
    Each A_block (2x2) times B_block (2x2) uses one Strassen decomposition (rank 7).
    Total: 2 A-blocks * 4 B-col-blocks * 2 B-row-blocks = 8 Strassen instances
    Total rank: 8 * 7 = 56
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank))  # 8 x 56
    V = np.zeros((m * p, rank))  # 32 x 56
    W = np.zeros((p * n, rank))  # 16 x 56
    
    # Strassen 2x2x2 decomposition
    # For A = [[a00, a01], [a10, a11]] and B = [[b00, b01], [b10, b11]]:
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
    
    # Define the 7 Strassen products:
    # Each is (a_factors, b_factors) where factors are lists of (element_idx, coeff)
    # element_idx: 0=a00/b00, 1=a01/b01, 2=a10/b10, 3=a11/b11
    strassen_products = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P1
        ([(2, 1), (3, 1)], [(0, 1)]),           # P2
        ([(0, 1)], [(1, 1), (3, -1)]),          # P3
        ([(3, 1)], [(2, 1), (0, -1)]),          # P4
        ([(0, 1), (1, 1)], [(3, 1)]),           # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P7
    ]
    
    # C reconstruction: for each C element, list of (product_idx, coeff)
    c_reconstruction = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00
        [(2, 1), (4, 1)],                   # C01
        [(1, 1), (3, 1)],                   # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)],  # C11
    ]
    
    # Define the 8 Strassen instances:
    # (a_col_off, b_row_off, b_col_off)
    # a_col_off: 0 for A_L, 2 for A_R
    # b_row_off: 0 for top 2 rows of B, 2 for bottom 2 rows
    # b_col_off: 0, 2, 4, 6 (which 2-col block of B)
    instances = [
        (0, 0, 0),   # A_L * B[0, 0:2]
        (0, 0, 2),   # A_L * B[0, 2:4]
        (0, 0, 4),   # A_L * B[0, 4:6]
        (0, 0, 6),   # A_L * B[0, 6:8]
        (2, 2, 0),   # A_R * B[1, 0:2]
        (2, 2, 2),   # A_R * B[1, 2:4]
        (2, 2, 4),   # A_R * B[1, 4:6]
        (2, 2, 6),   # A_R * B[1, 6:8]
    ]
    
    col = 0
    for inst_idx, (a_col_off, b_row_off, b_col_off) in enumerate(instances):
        for prod_idx in range(7):
            a_factors, b_factors = strassen_products[prod_idx]
            
            # Fill U (A factors)
            for a_elem, a_coeff in a_factors:
                a_row = a_elem // 2
                a_col = a_elem % 2
                u_idx = a_row * m + (a_col_off + a_col)
                U[u_idx, col] += a_coeff
            
            # Fill V (B factors)
            for b_elem, b_coeff in b_factors:
                b_row = b_elem // 2
                b_col = b_elem % 2
                v_idx = (b_row_off + b_row) * p + (b_col_off + b_col)
                V[v_idx, col] += b_coeff
            
            col += 1
    
    # Fill W (C reconstruction)
    col = 0
    for inst_idx, (a_col_off, b_row_off, b_col_off) in enumerate(instances):
        for c_elem in range(4):
            c_row = c_elem // 2
            c_col = c_elem % 2
            # W index: k*n + i where k = b_col_off + c_col, i = c_row
            w_idx = (b_col_off + c_col) * n + c_row
            for prod_idx, coeff in c_reconstruction[c_elem]:
                W[w_idx, col + prod_idx] += coeff
        col += 7
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
