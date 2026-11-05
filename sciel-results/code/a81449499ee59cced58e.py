import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = 30
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # Term 1: (a00+a11)(b00+b11)
    # A: 00, 11
    # B: 00, 11
    # C: 00, 11
    terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # Coeffs are dicts: {(i,k): val, ...}
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # Term 2: (a10+a11)b00
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        # Term 3: a00(b01-b11)
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        # Term 4: a11(b10-b00)
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        # Term 5: (a00+a01)b11
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        # Term 6: (a10-a00)(b00+b01)
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        # Term 7: (a01-a11)(b10+b11)
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): -1}
        )
    ]
    
    # Fill U, V, W for the 4 blocks of 2x2
    for q in range(4):
        base_col = q * 7
        for t in range(7):
            col = base_col + t
            a_coeffs, b_coeffs, c_coeffs = terms[t]
            
            for (i_loc, k_loc), val in a_coeffs.items():
                i_glob = i_loc
                k_glob = k_loc
                u_row = i_glob * m + k_glob
                U[u_row, col] += val
                
            for (k_loc, j_loc), val in b_coeffs.items():
                k_glob = k_loc
                j_glob = 2 * q + j_loc
                v_row = k_glob * p + j_glob
                V[v_row, col] += val
                
            for (i_loc, j_loc), val in c_coeffs.items():
                i_glob = i_loc
                j_glob = 2 * q + j_loc
                w_row = j_glob * n + i_glob
                W[w_row, col] += val
                
    # Fill U, V, W for the 2x1x8 part (Rank 2)
    # Term 8: a02 * b2j -> c0j
    col = 28
    # A: (0, 2)
    U[0 * m + 2, col] = 1
    # B: (2, j) for all j
    for j in range(p):
        V[2 * p + j, col] = 1
    # C: (0, j) for all j
    for j in range(p):
        W[j * n + 0, col] = 1
        
    # Term 9: a12 * b2j -> c1j
    col = 29
    # A: (1, 2)
    U[1 * m + 2, col] = 1
    # B: (2, j) for all j
    for j in range(p):
        V[2 * p + j, col] = 1
    # C: (1, j) for all j
    for j in range(p):
        W[j * n + 1, col] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
