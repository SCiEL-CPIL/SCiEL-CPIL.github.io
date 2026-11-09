"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Directly construct the valid rank 28 Block Strassen decomposition.
    # This avoids the invalid floating-point outputs from gradient descent
    # which fail the integer/half-integer constraint check.
    
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
    
    # Indices in 2x2 block: 00, 01, 10, 11 -> 0, 1, 2, 3
    # U indices: i*m + j. For 2x2 block: 0, 1, 2, 3
    # V indices: j*p + k. For 2x2 block: 0, 1, 2, 3
    # W indices: k*n + i. For 2x2 block: 0, 1, 2, 3
    
    strassen_terms = [
        # M1
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # M2
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        # M3
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        # M4
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        # M5
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        # M6
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # M7
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    # We are multiplying 2x4 by 4x4.
    # We can view this as 4 independent 2x2 multiplications by splitting the 4 columns of A and 4 columns of B into pairs?
    # Actually, A is 2x4, B is 4x4.
    # Let's partition A into two 2x2 blocks: A_left (cols 0,1), A_right (cols 2,3).
    # Let's partition B into two 2x2 blocks vertically? No, B is 4x4.
    # Let's partition B into four 2x2 blocks: B_tl (rows 0,1; cols 0,1), B_tr (rows 0,1; cols 2,3), B_bl (rows 2,3; cols 0,1), B_br (rows 2,3; cols 2,3).
    # 
    # C = A * B
    # C is 2x4.
    # C[:, 0:2] = A_left * B_tl + A_right * B_bl  (This is not quite right, dimensions don't match for simple block mult if we split A horizontally and B vertically like that).
    #
    # Let's use the standard block multiplication where we split the middle dimension (m=4) into two parts of size 2.
    # A = [A_1 A_2] where A_1 is 2x2 (cols 0,1) and A_2 is 2x2 (cols 2,3).
    # B = [B_1; B_2] where B_1 is 2x4 (rows 0,1) and B_2 is 2x4 (rows 2,3).
    # C = A_1 B_1 + A_2 B_2.
    #
    # Now B_1 is 2x4. We can split B_1 horizontally into B_11 (2x2, cols 0,1) and B_12 (2x2, cols 2,3).
    # B_2 is 2x4. We can split B_2 horizontally into B_21 (2x2, cols 0,1) and B_22 (2x2, cols 2,3).
    #
    # So C[:, 0:2] = A_1 B_11 + A_2 B_21.
    # And C[:, 2:4] = A_1 B_12 + A_2 B_22.
    #
    # This requires 4 separate 2x2 * 2x2 multiplications.
    # Each 2x2 * 2x2 multiplication can be done with rank 7 (Strassen).
    # Total rank = 4 * 7 = 28.
    
    # Block 1: A_1 (cols 0,1) * B_11 (rows 0,1, cols 0,1) -> contributes to C[:, 0:2]
    # A_1 indices: i in 0..1, j in 0..1. Global j = 0,1.
    # B_11 indices: j in 0..1, k in 0..1. Global j = 0,1. Global k = 0,1.
    # C contribution: k in 0..1, i in 0..1. Global k = 0,1. Global i = 0,1.
    
    # Block 2: A_2 (cols 2,3) * B_21 (rows 2,3, cols 0,1) -> contributes to C[:, 0:2]
    # A_2 indices: i in 0..1, j in 2..3. Global j = 2,3.
    # B_21 indices: j in 2..3, k in 0..1. Global j = 2,3. Global k = 0,1.
    # C contribution: k in 0..1, i in 0..1. Global k = 0,1. Global i = 0,1.
    
    # Block 3: A_1 (cols 0,1) * B_12 (rows 0,1, cols 2,3) -> contributes to C[:, 2:4]
    # A_1 indices: i in 0..1, j in 0..1. Global j = 0,1.
    # B_12 indices: j in 0..1, k in 2..3. Global j = 0,1. Global k = 2,3.
    # C contribution: k in 2..3, i in 0..1. Global k = 2,3. Global i = 0,1.
    
    # Block 4: A_2 (cols 2,3) * B_22 (rows 2,3, cols 2,3) -> contributes to C[:, 2:4]
    # A_2 indices: i in 0..1, j in 2..3. Global j = 2,3.
    # B_22 indices: j in 2..3, k in 2..3. Global j = 2,3. Global k = 2,3.
    # C contribution: k in 2..3, i in 0..1. Global k = 2,3. Global i = 0,1.
    
    blocks = [
        {'j_offset': 0, 'k_offset': 0}, # Block 1
        {'j_offset': 2, 'k_offset': 0}, # Block 2
        {'j_offset': 0, 'k_offset': 2}, # Block 3
        {'j_offset': 2, 'k_offset': 2}  # Block 4
    ]
    
    col = 0
    for block in blocks:
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j][col] += coeff
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k][col] += coeff
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i][col] += coeff
            col += 1
        
    # Ensure all values are integers (they are already, but explicit cast helps)
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
