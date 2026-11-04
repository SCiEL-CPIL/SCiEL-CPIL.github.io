import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        for (i, j), val in u_dict.items():
            U[i * m + j][col] += val
        for (j, k), val in v_dict.items():
            V[j * p + k][col] += val
        for (k, i), val in w_dict.items():
            W[k * n + i][col] += val
        col += 1

    # Strassen terms for 2x2x2
    # u: {(i,j): val}, v: {(j,k): val}, w: {(k,i): val}
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,1):1, (0,1):1}, {(0,0):1})
    ]
    
    # Schoolbook terms for 2x2x1
    # B is 2x1, so k is always 0 locally.
    schoolbook_terms = [
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        ({(0,1):1}, {(1,0):1}, {(0,0):1}),
        ({(1,0):1}, {(0,0):1}, {(0,1):1}),
        ({(1,1):1}, {(1,0):1}, {(0,1):1})
    ]

    # Process 8 blocks
    # A_1 (j 0-1), A_2 (j 2-3)
    # B_1 (j 0-1), B_2 (j 2-3)
    # B blocks k: 0-1, 2-3, 4-5, 6
    
    blocks = [
        # (j_offset_A, j_offset_B, k_offset, is_schoolbook)
        (0, 0, 0, False), # A1 B1a
        (0, 0, 2, False), # A1 B1b
        (0, 0, 4, False), # A1 B1c
        (0, 0, 6, True),  # A1 B1d
        (2, 2, 0, False), # A2 B2a
        (2, 2, 2, False), # A2 B2b
        (2, 2, 4, False), # A2 B2c
        (2, 2, 6, True)   # A2 B2d
    ]
    
    for j_off_A, j_off_B, k_off, is_schoolbook in blocks:
        terms = schoolbook_terms if is_schoolbook else strassen_terms
        for u_dict, v_dict, w_dict in terms:
            # Map local indices to global
            u_glob = {}
            for (i, j), val in u_dict.items():
                u_glob[(i, j_off_A + j)] = val
            
            v_glob = {}
            for (j, k), val in v_dict.items():
                v_glob[(j_off_B + j, k_off + k)] = val
            
            w_glob = {}
            for (k, i), val in w_dict.items():
                w_glob[(k_off + k, i)] = val
                
            add_term(u_glob, v_glob, w_glob)

    return {"U": U, "V": V, "W": W}
