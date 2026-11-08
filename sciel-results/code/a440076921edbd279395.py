import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    rank = 28
    
    # Initialize matrices
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (32, 28)
    # W: (p*n, rank) = (16, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 * 2x2
    # Each term is a dict of (local_row, local_col) -> coeff for U, V, and W
    # U corresponds to A (2x2)
    # V corresponds to B (2x2)
    # W corresponds to C (2x2)
    
    strassen_terms = [
        # 1
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        # 2
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1, (1,1): 1}},
        # 3
        {'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): 1}},
        # 4
        {'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (1,0): 1}},
        # 5
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): 1, (0,1): -1}},
        # 6
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}},
        # 7
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}},
    ]
    
    # Block definitions:
    # (a_row_offset, a_col_offset, b_row_offset, b_col_offset)
    blocks = [
        (0, 0, 0, 0),   # A_L, B_T1
        (0, 0, 0, 2),   # A_L, B_T2
        (0, 2, 2, 0),   # A_R, B_B1
        (0, 2, 2, 2),   # A_R, B_B2
    ]
    
    col = 0
    for a_ro, a_co, b_ro, b_co in blocks:
        for term in strassen_terms:
            # Fill U
            for (lr, lc), coeff in term['U'].items():
                global_i = lr
                global_j = lc + a_co
                # U index: i * m + j
                u_idx = global_i * m + global_j
                U[u_idx][col] += coeff
                
            # Fill V
            for (lr, lc), coeff in term['V'].items():
                global_j = lr + b_ro
                global_k = lc + b_co
                # V index: j * p + k
                v_idx = global_j * p + global_k
                V[v_idx][col] += coeff
                
            # Fill W
            for (lr, lc), coeff in term['W'].items():
                global_i = lr
                global_k = lc
                # W index: k * n + i
                w_idx = global_k * n + global_i
                W[w_idx][col] += coeff
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
