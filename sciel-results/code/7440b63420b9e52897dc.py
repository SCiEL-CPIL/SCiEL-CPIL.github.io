"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Each term is a dict: {'A': {idx: val}, 'B': {idx: val}, 'C': {idx: val}}
    # Local indices 0,1 for i,j,k
    strassen_terms = [
        # M1
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        # M2
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
        # M3
        {'A': {(0,0): 1}, 'B': {(1,0): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
        # M4
        {'A': {(1,1): 1}, 'B': {(0,1): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        # M5
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        # M6
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        # M7
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}},
    ]
    
    # Define the 4 sub-blocks
    # (a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_k_offset, c_i_offset)
    # Note: A is 2x4, B is 4x4, C is 2x4
    # A indices: i in [0,2), j in [0,4)
    # B indices: j in [0,4), k in [0,4)
    # C indices: i in [0,2), k in [0,4) -> W index is k*n + i
    
    blocks = [
        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        {'a_i': 0, 'a_j': 0, 'b_j': 0, 'b_k': 0, 'c_k': 0, 'c_i': 0},
        # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        {'a_i': 0, 'a_j': 2, 'b_j': 2, 'b_k': 0, 'c_k': 0, 'c_i': 0},
        # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        {'a_i': 0, 'a_j': 0, 'b_j': 0, 'b_k': 2, 'c_k': 2, 'c_i': 0},
        # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        {'a_i': 0, 'a_j': 2, 'b_j': 2, 'b_k': 2, 'c_k': 2, 'c_i': 0},
    ]
    
    current_rank = 0
    for block in blocks:
        for term in strassen_terms:
            # Populate U
            for (li, lj), val in term['A'].items():
                gi = block['a_i'] + li
                gj = block['a_j'] + lj
                u_idx = gi * m + gj
                U[u_idx][current_rank] = val
            
            # Populate V
            for (lj, lk), val in term['B'].items():
                gj = block['b_j'] + lj
                gk = block['b_k'] + lk
                v_idx = gj * p + gk
                V[v_idx][current_rank] = val
            
            # Populate W
            for (lk, li), val in term['C'].items():
                gk = block['c_k'] + lk
                gi = block['c_i'] + li
                w_idx = gk * n + gi
                W[w_idx][current_rank] = val
            
            current_rank += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
