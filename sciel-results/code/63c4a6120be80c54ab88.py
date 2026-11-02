import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Target rank 46 decomposition using 3+2 split for B columns/rows?
    # Actually, the 46 rank decomposition used:
    # A split: 2, 2, 1 (cols)
    # B split: rows 2, 2, 1; cols 2, 2, 1
    # This resulted in 4 blocks of M_2,2,2 (rank 7) and others.
    
    # Let's build the 46 rank decomposition.
    # U: 10 x 46
    # V: 25 x 46
    # W: 10 x 46
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col_idx
        if col_idx >= rank:
            return
        U[u_idx][col_idx] += val
        V[v_idx][col_idx] += val
        W[w_idx][col_idx] += val
        col_idx += 1

    # Strassen 2x2x2 decomposition terms
    # Inputs: A (2x2), B (2x2)
    # A indices: 0,1 (rows), 0,1 (cols) -> A_idx = r*2+c
    # B indices: 0,1 (rows), 0,1 (cols) -> B_idx = r*2+c
    # C indices: 0,1 (rows), 0,1 (cols) -> C_idx = r*2+c
    
    def add_strassen_222(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        # A is at [a_row_offset, a_row_offset+2) x [a_col_offset, a_col_offset+2)
        # B is at [b_row_offset, b_row_offset+2) x [b_col_offset, b_col_offset+2)
        # C is at [c_row_offset, c_row_offset+2) x [c_col_offset, c_col_offset+2)
        
        # P1 = (A00+A11)(B00+B11)
        # A00: a_row_offset, a_col_offset
        # A11: a_row_offset+1, a_col_offset+1
        # B00: b_row_offset, b_col_offset
        # B11: b_row_offset+1, b_col_offset+1
        
        # We need to add terms to U, V, W.
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # This function adds 7 terms.
        
        # P1
        # U: A00, A11
        # V: B00, B11
        # W: C00, C01, C10, C11 (with signs)
        
        # Actually, it's easier to just list the 7 rank-1 tensors.
        # Each rank-1 tensor is u v w.
        # u has non-zeros at A00, A11.
        # v has non-zeros at B00, B11.
        # w has non-zeros at C00, C01, C10, C11.
        
        # This is complex to hardcode generally.
        # Instead, I will use the schoolbook for the 2x2x2 blocks?
        # Schoolbook rank is 4. Strassen is 7.
        # Wait, 7 > 4.
        # So using Strassen INCREASES the rank for 2x2x2!
        # Schoolbook 2x2x2 is rank 4.
        # My previous calculation:
        # 4 blocks of 2x2x2.
        # If I use schoolbook (rank 4), total rank is 4*4 + 18 = 34.
        # If I use Strassen (rank 7), total rank is 4*7 + 18 = 46.
        # So I should use SCHOOLBOOK for the 2x2x2 blocks!
        
        pass

    # Let's restart with Schoolbook for all sub-blocks.
    # Rank 34 decomposition.
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col_idx
        if col_idx >= rank:
            return
        U[u_idx][col_idx] += val
        V[v_idx][col_idx] += val
        W[w_idx][col_idx] += val
        col_idx += 1

    # A split: 2, 2, 1 (cols 0-1, 2-3, 4)
    # B split: rows 2,2,1 (0-1, 2-3, 4); cols 2,2,1 (0-1, 2-3, 4)
    
    # Block A1 (2x2) * B11 (2x2)
    # A1: rows 0-1, cols 0-1
    # B11: rows 0-1, cols 0-1
    # C11: rows 0-1, cols 0-1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                add_term(i*5 + j, j*5 + k, k*2 + i)
                
    # Block A1 (2x2) * B21 (2x2)
    # A1: rows 0-1, cols 0-1
    # B21: rows 2-3, cols 0-1
    # C11: rows 0-1, cols 0-1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A index: i, j (col offset 0)
                # B index: j+2, k (row offset 2)
                # C index: i, k
                add_term(i*5 + j, (j+2)*5 + k, k*2 + i)

    # Block A1 (2x2) * B31 (1x2)
    # A1: rows 0-1, cols 0-1
    # B31: row 4, cols 0-1
    # C11: rows 0-1, cols 0-1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A index: i, j
                # B index: 4, k
                # C index: i, k
                add_term(i*5 + j, 4*5 + k, k*2 + i)

    # Block A2 (2x2) * B12 (2x2)
    # A2: rows 0-1, cols 2-3
    # B12: rows 0-1, cols 2-3
    # C12: rows 0-1, cols 2-3
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A index: i, j+2
                # B index: j, k+2
                # C index: i, k+2
                add_term(i*5 + (j+2), j*5 + (k+2), (k+2)*2 + i)

    # Block A2 (2x2) * B22 (2x2)
    # A2: rows 0-1, cols 2-3
    # B22: rows 2-3, cols 2-3
    # C12: rows 0-1, cols 2-3
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A index: i, j+2
                # B index: j+2, k+2
                # C index: i, k+2
                add_term(i*5 + (j+2), (j+2)*5 + (k+2), (k+2)*2 + i)

    # Block A2 (2x2) * B32 (1x2)
    # A2: rows 0-1, cols 2-3
    # B32: row 4, cols 2-3
    # C12: rows 0-1, cols 2-3
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A index: i, j+2
                # B index: 4, k+2
                # C index: i, k+2
                add_term(i*5 + (j+2), 4*5 + (k+2), (k+2)*2 + i)

    # Block A3 (2x1) * B13 (2x1)
    # A3: rows 0-1, col 4
    # B13: rows 0-1, col 4
    # C13: rows 0-1, col 4
    for i in range(2):
        for k in range(1):
            # j is fixed at 4?
            # A index: i, 4
            # B index: j, 4 (j from 0 to 1)
            # C index: i, 4
            for j in range(2):
                add_term(i*5 + 4, j*5 + 4, 4*2 + i)

    # Block A3 (2x1) * B23 (2x1)
    # A3: rows 0-1, col 4
    # B23: rows 2-3, col 4
    # C13: rows 0-1, col 4
    for i in range(2):
        for k in range(1):
            for j in range(2):
                add_term(i*5 + 4, (j+2)*5 + 4, 4*2 + i)

    # Block A3 (2x1) * B33 (1x1)
    # A3: rows 0-1, col 4
    # B33: row 4, col 4
    # C13: rows 0-1, col 4
    for i in range(2):
        for k in range(1):
            # j is 4
            add_term(i*5 + 4, 4*5 + 4, 4*2 + i)

    return {"U": U, "V": V, "W": W}
