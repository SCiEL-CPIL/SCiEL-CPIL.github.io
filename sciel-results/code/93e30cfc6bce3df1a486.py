import time
import numpy as np

def construct():
    """
    Rank-42 exact decomposition for 2x4 @ 4x6 matrix multiplication.
    
    Strategy: Split the 4x4 middle dimension into two 2x2 blocks (left and right halves).
    This gives C = A_left @ B_top + A_right @ B_bottom, where each term is a 2x2 @ 2x6 multiplication.
    Each 2x2 @ 2x6 is decomposed into three 2x2 @ 2x2 multiplications using Strassen (rank 7 each).
    Total rank = 2 * 3 * 7 = 42.
    
    U: (n*m, rank) = (8, 42)
    V: (m*p, rank) = (24, 42)
    W: (p*n, rank) = (12, 42)
    
    W index convention: W[k*n+i, r] where k is column index (0..5), i is row index (0..1)
    So W index = k*2 + i for output C[i][k]
    """
    n, m, p = 2, 4, 6
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=int)  # 8 x 42
    V = np.zeros((m * p, rank), dtype=int)  # 24 x 42
    W = np.zeros((p * n, rank), dtype=int)  # 12 x 42
    
    col = 0
    
    # Strassen 2x2x2 core: 7 multiplications
    # For X @ Y where X,Y are 2x2:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    # Result:
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    # We decompose 2x4 @ 4x6 as:
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    # Each term is 2x2 @ 2x6
    # Split 2x6 into three 2x2 blocks: cols 0-1, 2-3, 4-5
    # So 2x2 @ 2x6 = concat of three 2x2 @ 2x2 multiplications
    # Each 2x2 @ 2x2 uses Strassen rank 7
    # Total: 2 blocks * 3 sub-blocks * 7 = 42
    
    # Define Strassen pattern for 2x2 @ 2x2
    # Each term: (list of (a_row, a_col, sign) for A, list of (b_row, b_col, sign) for B, list of (out_row, out_col, sign) for output)
    strassen_terms = [
        # M1: (A00+A11)*(B00+B11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2: (A10+A11)*B00
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3: A00*(B01-B11)
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4: A11*(B10-B00)
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5: (A00+A01)*B11
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6: (A10-A00)*(B00+B01)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7: (A01-A11)*(B10+B11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    # Helper: apply Strassen for a 2x2 @ 2x2 sub-problem
    # a_base: (row_offset, col_offset) in A (2x4 matrix, indexed as [i][j])
    # b_base: (row_offset, col_offset) in B (4x6 matrix, indexed as [i][j])
    # out_base: (row_offset, col_offset) in output C (2x6, indexed as [i][j])
    def apply_strassen(a_base, b_base, out_base):
        nonlocal col
        for a_terms, b_terms, w_terms in strassen_terms:
            # Set U coefficients
            for (ar, ac, sign) in a_terms:
                u_idx = (a_base[0] + ar) * m + (a_base[1] + ac)
                U[u_idx, col] += sign
            
            # Set V coefficients
            for (br, bc, sign) in b_terms:
                v_idx = (b_base[0] + br) * p + (b_base[1] + bc)
                V[v_idx, col] += sign
            
            # Set W coefficients
            for (or_, oc, sign) in w_terms:
                w_idx = (out_base[1] + oc) * n + (out_base[0] + or_)
                W[w_idx, col] += sign
            
            col += 1
    
    # Left half: A[:,0:2] @ B[0:2,:]
    # Split into 3 sub-blocks of B: cols 0-1, 2-3, 4-5
    for sub in range(3):
        b_col_offset = 2 * sub
        out_col_offset = 2 * sub
        apply_strassen((0, 0), (0, b_col_offset), (0, out_col_offset))
    
    # Right half: A[:,2:4] @ B[2:4,:]
    # Split into 3 sub-blocks of B: cols 0-1, 2-3, 4-5
    for sub in range(3):
        b_col_offset = 2 * sub
        out_col_offset = 2 * sub
        apply_strassen((0, 2), (2, b_col_offset), (0, out_col_offset))
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
