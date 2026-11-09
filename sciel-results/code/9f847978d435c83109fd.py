import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    rank = 28
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen rank-7 decomposition for 2x2x2 tensor
    # Using 0-indexed: a00, a01, a10, a11 (row, col)
    # b00, b01, b10, b11
    # c00, c01, c10, c11
    #
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = (a00-a11)(b10+b11)
    # M4 = a00(b01-b11)
    # M5 = a11(b10-b00)
    # M6 = (a00+a01)b11
    # M7 = (a10-a01)(b00+b01)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M6 + M7
    # c10 = M2 - M4
    # c11 = M1 - M2 + M3 + M5
    
    strassen_terms = [
        # Term 1: (a00+a11)(b00+b11)
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        
        # Term 2: (a10+a11)b00
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        
        # Term 3: a00(b01-b11)
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        
        # Term 4: a11(b10-b00)
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        
        # Term 5: (a00+a01)b11
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        
        # Term 6: (a10-a01)(b00+b01)
        {'U': [(2, 1), (1, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        
        # Term 7: (a01-a11)(b10+b11)
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]},
    ]
    
    # The 4 blocks for the 2x4 by 4x4 multiplication:
    # The A matrix is 2x4 (rows: 0,1; cols: 0,1,2,3)
    # The B matrix is 4x4 (rows: 0,1,2,3; cols: 0,1,2,3)
    # The C matrix is 2x4 (rows: 0,1; cols: 0,1,2,3)
    #
    # We split the intermediate dimension (m=4) into two groups of 2,
    # and the output columns (p=4) into two groups of 2.
    #
    # For each (j_block, k_block) pair where j_block in {0,1} and k_block in {0,1},
    # we have a 2x2 sub-multiplication:
    # A[:, j_block*2:(j_block+1)*2] * B[j_block*2:(j_block+1)*2, k_block*2:(k_block+1)*2] -> C[:, k_block*2:(k_block+1)*2]
    #
    # Each such sub-multiplication is a 2x2x2 tensor multiplication, which can be done
    # with Strassen rank 7.
    
    blocks = [
        {'j_offset': 0, 'k_offset': 0},
        {'j_offset': 0, 'k_offset': 2},
        {'j_offset': 2, 'k_offset': 0},
        {'j_offset': 2, 'k_offset': 2},
    ]
    
    col = 0
    for block in blocks:
        if time.time() > deadline:
            break
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            if time.time() > deadline:
                break
            # Add to U
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j][col] += coeff
            
            # Add to V
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k][col] += coeff
            
            # Add to W
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
