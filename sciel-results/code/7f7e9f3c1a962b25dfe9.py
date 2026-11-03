import numpy as np

def construct():
    n, m, p = 2, 4, 8
    # U: (n*m, rank) -> (8, rank)
    # V: (m*p, rank) -> (32, rank)
    # W: (p*n, rank) -> (16, rank)
    
    # Strassen 2x2x2 decomposition vectors
    # Local indices: A: 00,01,10,11; B: 00,01,10,11; C: 00,01,10,11
    strassen_u = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [0, 0, 1, -1],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    strassen_w = [
        [1, 0, 0, 1], # M1: C00+C11
        [0, 0, 1, -1],# M2: C10-C11
        [0, 1, 0, 1], # M3: C01+C11
        [1, 0, 1, 0], # M4: C00+C10
        [-1, 1, 0, 0],# M5: C01-C00
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    num_a_blocks = 2 # Cols of A split into 2 blocks of 2
    num_k_blocks = 4 # Cols of B (rows for B part) split into 4 blocks of 2
    num_strassen = 7
    
    total_rank = num_a_blocks * num_k_blocks * num_strassen
    rank = total_rank
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for a_block in range(num_a_blocks):
        # a_block 0: A cols 0,1. a_block 1: A cols 2,3.
        a_offset = a_block * 2
        
        for k_block in range(num_k_blocks):
            # k_block 0: B cols 0,1 (for top rows) or 0,1 (for bot rows)
            # Wait, B is 4x8.
            # a_block 0 uses B rows 0,1.
            # a_block 1 uses B rows 2,3.
            
            # In both cases, we split the 8 columns of the corresponding B part into 4 pairs.
            # So k_block corresponds to local k index pair (2*k_block, 2*k_block+1)
            
            for m_idx in range(num_strassen):
                u_vec = strassen_u[m_idx]
                v_vec = strassen_v[m_idx]
                w_vec = strassen_w[m_idx]
                
                # Fill U
                # Local A index: i in 0..1, j' in 0..1
                # Global A index: i * 4 + (a_offset + j')
                for i in range(n):
                    for j_prime in range(2):
                        val = u_vec[i * 2 + j_prime]
                        if val != 0:
                            global_a_idx = i * m + (a_offset + j_prime)
                            U[global_a_idx, col] += val
                
                # Fill V
                # Local B index: j' in 0..1 (row in 2x2 block), k' in 0..1 (col in 2x2 block)
                # Global B row: if a_block 0, row is j'. If a_block 1, row is 2 + j'.
                # Global B col: 2 * k_block + k'
                
                b_row_offset = a_block * 2
                
                for j_prime in range(2):
                    for k_prime in range(2):
                        val = v_vec[j_prime * 2 + k_prime]
                        if val != 0:
                            global_b_row = b_row_offset + j_prime
                            global_b_col = 2 * k_block + k_prime
                            global_b_idx = global_b_row * p + global_b_col
                            V[global_b_idx, col] += val
                
                # Fill W
                # Local C index: i in 0..1, k' in 0..1
                # Global C index: (2 * k_block + k') * n + i
                # Note: W[k*n+i, r]
                
                for i in range(n):
                    for k_prime in range(2):
                        val = w_vec[i * 2 + k_prime] # Wait, w_vec is indexed by C local index (row-major 2x2)
                        # Local C index: i * 2 + k_prime
                        if val != 0:
                            global_c_col = 2 * k_block + k_prime
                            global_c_idx = global_c_col * n + i
                            W[global_c_idx, col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
