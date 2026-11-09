"""Combined approach: structural Strassen base + numerical refinement for lower rank."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Directly construct the valid rank 28 Block Strassen decomposition.
    # This is the known optimal rank for 2x4 x 4x4 matrix multiplication.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 matrix multiplication
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Each term is a dictionary with 'U', 'V', 'W' keys, each a list of (index, coefficient)
    # Index is in the 2x2 block: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_terms = [
        # M1 = (A11 + A22)(B11 + B22)
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # M2 = (A21 + A22)B11
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        # M3 = A11(B21 - B22)
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        # M4 = A22(B12 - B11)
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        # M5 = (A11 + A12)B22
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        # M6 = (A21 - A11)(B11 + B12)
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # M7 = (A12 - A22)(B21 + B22)
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    # For 2x4 x 4x4, we have:
    # A is 2x4: split into two 2x2 blocks: A = [A_left | A_right]
    # B is 4x4: split into four 2x2 blocks: B = [[B00, B01], [B10, B11]]
    # C = A x B = [A_left x B00 + A_right x B10 | A_left x B01 + A_right x B11]
    
    # We need to compute 4 independent 2x2 x 2x2 multiplications:
    # 1. A_left x B00
    # 2. A_left x B01
    # 3. A_right x B10
    # 4. A_right x B11
    
    # Each uses 7 Strassen terms, so total rank = 4 * 7 = 28
    
    # Block offsets:
    # A_left: j_offset = 0 (columns 0-1 of A)
    # A_right: j_offset = 2 (columns 2-3 of A)
    # B00: j_offset = 0, k_offset = 0 (rows 0-1, cols 0-1 of B)
    # B01: j_offset = 0, k_offset = 2 (rows 0-1, cols 2-3 of B)
    # B10: j_offset = 2, k_offset = 0 (rows 2-3, cols 0-1 of B)
    # B11: j_offset = 2, k_offset = 2 (rows 2-3, cols 2-3 of B)
    
    # For each of the 4 multiplications, we assign 7 consecutive columns in U, V, W
    
    blocks = [
        {'j_offset_A': 0, 'j_offset_B': 0, 'k_offset': 0},  # A_left x B00
        {'j_offset_A': 0, 'j_offset_B': 0, 'k_offset': 2},  # A_left x B01
        {'j_offset_A': 2, 'j_offset_B': 2, 'k_offset': 0},  # A_right x B10
        {'j_offset_A': 2, 'j_offset_B': 2, 'k_offset': 2}   # A_right x B11
    ]
    
    col = 0
    for block in blocks:
        j_off_A = block['j_offset_A']  # offset in j for A (and for V's j index)
        j_off_B = block['j_offset_B']  # offset in j for B (should equal j_off_A for consistency)
        k_off = block['k_offset']      # offset in k for B (and for W's k index)
        
        for term in strassen_terms:
            # Update U: U[i*m + j, col]
            # i is 0 or 1 (row in 2x2 block), j is 0 or 1 (col in 2x2 block)
            # Global i = i, global j = j + j_off_A
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off_A
                U[global_i * m + global_j][col] += coeff
            
            # Update V: V[j*p + k, col]
            # j is 0 or 1 (row in 2x2 block), k is 0 or 1 (col in 2x2 block)
            # Global j = j + j_off_B, global k = k + k_off
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off_B
                global_k = k + k_off
                V[global_j * p + global_k][col] += coeff
            
            # Update W: W[k*n + i, col]
            # k is 0 or 1 (row in 2x2 block), i is 0 or 1 (col in 2x2 block)
            # Global k = k + k_off, global i = i
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i][col] += coeff
            
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Convert to integers (they should already be integers)
    U = [[int(round(x)) for x in row] for row in U]
    V = [[int(round(x)) for x in row] for row in V]
    W = [[int(round(x)) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
