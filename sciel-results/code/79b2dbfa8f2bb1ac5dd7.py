import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    total_rank = 42
    
    # Initialize matrices with zeros
    mat_u = np.zeros((n * m, total_rank))
    mat_v = np.zeros((m * p, total_rank))
    mat_w = np.zeros((p * n, total_rank))
    
    # Strassen coefficients defined as dictionaries for sparse access
    # Indices (row, col) for 2x2 local blocks
    u_coefficients = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1},
    ]
    
    v_coefficients = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1},
    ]
    
    w_coefficients = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]
    
    current_col = 0
    
    # Iterate over blocks
    # a_block: 0, 1 (splits m=4 into two 2s)
    # b_block: 0, 1, 2 (splits p=6 into three 2s)
    for a_blk in range(2):
        for b_blk in range(3):
            for term_idx in range(7):
                if time.time() - start_time > 110:
                    break
                
                # Populate U
                for (r, c), coef in u_coefficients[term_idx].items():
                    row_u = r
                    col_u = a_blk * 2 + c
                    mat_u[row_u * m + col_u, current_col] += coef
                
                # Populate V
                for (r, c), coef in v_coefficients[term_idx].items():
                    row_v = a_blk * 2 + r
                    col_v = b_blk * 2 + c
                    mat_v[row_v * p + col_v, current_col] += coef
                
                # Populate W (Trace convention: W[k*n + i])
                for (r, c), coef in w_coefficients[term_idx].items():
                    row_w = r
                    col_w = b_blk * 2 + c
                    mat_w[col_w * n + row_w, current_col] += coef
                
                current_col += 1
            
            if time.time() - start_time > 110:
                break
        
        if time.time() - start_time > 110:
            break
    
    # Convert to lists of integers if possible
    def to_list_of_ints(arr):
        return [[int(val) if float(val).is_integer() else val for val in row] for row in arr.tolist()]

    return {
        "U": to_list_of_ints(mat_u),
        "V": to_list_of_ints(mat_v),
        "W": to_list_of_ints(mat_w)
    }
