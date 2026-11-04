import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Part 1: 4x4x4 using 8 Strassen 2x2x2 blocks (rank 56)
    # Part 2: 4x4x5 using Schoolbook (rank 80)
    
    rank1 = 56
    rank2 = 80
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen 2x2x2 terms
    # M1: (0,0,0,1), (3,0,0,1), (0,3,0,1), (3,3,0,1) -> C0
    # M2: (2,0,0,1), (3,0,0,1) -> C2
    # M3: (0,1,0,1), (0,3,0,-1) -> C1
    # M4: (3,2,0,1), (3,0,0,-1) -> C0, C2
    # M5: (0,3,0,1), (1,3,0,1) -> C0, C1
    # M6: (2,0,0,1), (0,0,0,-1), (2,1,0,1), (0,1,0,-1) -> C3
    # M7: (1,2,0,1), (3,2,0,-1), (1,3,0,1), (3,3,0,-1) -> C0, C3
    
    # We will define the 7 terms for one 2x2x2 block
    # Each term is a list of (u_idx, v_idx, w_idx, coeff)
    
    strassen_2x2x2 = [
        [(0,0,0,1), (3,0,0,1), (0,3,0,1), (3,3,0,1)], # M1
        [(2,0,0,1), (3,0,0,1)], # M2
        [(0,1,0,1), (0,3,0,-1)], # M3
        [(3,2,0,1), (3,0,0,-1)], # M4
        [(0,3,0,1), (1,3,0,1)], # M5
        [(2,0,0,1), (0,0,0,-1), (2,1,0,1), (0,1,0,-1)], # M6
        [(1,2,0,1), (3,2,0,-1), (1,3,0,1), (3,3,0,-1)] # M7
    ]
    
    # C0 = M1 + M4 - M5 + M7
    # C1 = M3 + M5
    # C2 = M2 + M4
    # C3 = M1 - M2 + M3 + M6
    
    # We need to map these to the 4x4x4 blocks.
    # A11 is rows 0-3, cols 0-3 of A (in 4x4)
    # Actually, A is 4x4.
    # A11 is A[0:2, 0:2] -> indices 0,1,4,5
    # A12 is A[0:2, 2:4] -> indices 2,3,6,7
    # A21 is A[2:4, 0:2] -> indices 8,9,12,13
    # A22 is A[2:4, 2:4] -> indices 10,11,14,15
    
    # B11 is B[0:2, 0:2] -> indices 0,1,4,5
    # B12 is B[0:2, 2:4] -> indices 2,3,6,7
    # B21 is B[2:4, 0:2] -> indices 8,9,12,13
    # B22 is B[2:4, 2:4] -> indices 10,11,14,15
    
    # C11 is C[0:2, 0:2] -> indices 0,1,4,5
    # C12 is C[0:2, 2:4] -> indices 2,3,6,7
    # C21 is C[2:4, 0:2] -> indices 8,9,12,13
    # C22 is C[2:4, 2:4] -> indices 10,11,14,15
    
    # We will add the 8 blocks.
    # Block 1: A11 * B11 -> C11
    # Block 2: A12 * B21 -> C11
    # Block 3: A11 * B12 -> C12
    # Block 4: A12 * B22 -> C12
    # Block 5: A21 * B11 -> C21
    # Block 6: A22 * B21 -> C21
    # Block 7: A21 * B12 -> C22
    # Block 8: A22 * B22 -> C22
    
    # For each block, we add 7 terms.
    # Total 56 terms.
    
    col = 0
    for block in range(8):
        # Determine A block, B block, C block
        if block == 0:
            a_block = 0 # A11
            b_block = 0 # B11
            c_block = 0 # C11
        elif block == 1:
            a_block = 1 # A12
            b_block = 2 # B21
            c_block = 0 # C11
        elif block == 2:
            a_block = 0 # A11
            b_block = 1 # B12
            c_block = 1 # C12
        elif block == 3:
            a_block = 1 # A12
            b_block = 3 # B22
            c_block = 1 # C12
        elif block == 4:
            a_block = 2 # A21
            b_block = 0 # B11
            c_block = 2 # C21
        elif block == 5:
            a_block = 3 # A22
            b_block = 2 # B21
            c_block = 2 # C21
        elif block == 6:
            a_block = 2 # A21
            b_block = 1 # B12
            c_block = 3 # C22
        elif block == 7:
            a_block = 3 # A22
            b_block = 3 # B22
            c_block = 3 # C22
            
        # Map block indices to actual indices
        # A block 0: 0,1,4,5
        # A block 1: 2,3,6,7
        # A block 2: 8,9,12,13
        # A block 3: 10,11,14,15
        
        a_map = [0,1,4,5]
        if a_block == 1: a_map = [2,3,6,7]
        if a_block == 2: a_map = [8,9,12,13]
        if a_block == 3: a_map = [10,11,14,15]
        
        b_map = [0,1,4,5]
        if b_block == 1: b_map = [2,3,6,7]
        if b_block == 2: b_map = [8,9,12,13]
        if b_block == 3: b_map = [10,11,14,15]
        
        c_map = [0,1,4,5]
        if c_block == 1: c_map = [2,3,6,7]
        if c_block == 2: c_map = [8,9,12,13]
        if c_block == 3: c_map = [10,11,14,15]
        
        # Add the 7 Strassen terms for this block
        for term in strassen_2x2x2:
            for (u, v, w, coeff) in term:
                U[a_map[u]][col] += coeff
                V[b_map[v]][col] += coeff
                W[c_map[w]][col] += coeff
            col += 1
            
    # Part 2: 4x4x5 Schoolbook
    # A is 4x4, B is 4x5, C is 4x5
    # B indices 16..35
    # C indices 16..35
    
    for i in range(4):
        for j in range(4):
            for k in range(5):
                U[i * 4 + j][col] = 1
                V[j * 5 + k + 16][col] = 1
                W[k * 4 + i + 16][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
