import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen terms for 2x2x2
    # Each term: (list of (row, col, coeff) for A, list for B, list for C)
    strassen_terms = [
        # M1
        {
            'A': [(0, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1), (1, 1, 1)]
        },
        # M2
        {
            'A': [(1, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1)],
            'C': [(1, 0, 1), (1, 1, -1)]
        },
        # M3
        {
            'A': [(0, 0, 1)],
            'B': [(1, 0, 1), (1, 1, -1)],
            'C': [(0, 1, 1), (1, 1, 1)]
        },
        # M4
        {
            'A': [(1, 1, 1)],
            'B': [(0, 1, 1), (0, 0, -1)],
            'C': [(0, 0, -1), (1, 0, 1)]
        },
        # M5
        {
            'A': [(0, 0, 1), (0, 1, 1)],
            'B': [(1, 1, 1)],
            'C': [(0, 0, 1), (0, 1, 1)]
        },
        # M6
        {
            'A': [(1, 0, 1), (0, 0, -1)],
            'B': [(0, 0, 1), (0, 1, 1)],
            'C': [(1, 1, 1)]
        },
        # M7
        {
            'A': [(0, 1, 1), (1, 1, -1)],
            'B': [(1, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1)]
        }
    ]
    
    col = 0
    # j is the block column index for B and C (0, 1, 2)
    for j in range(3):
        # Part 1: A1 * B1j
        # A1 is A[:, 0:2]
        # B1j is B[0:2, 2*j : 2*j+2]
        # Output is C[:, 2*j : 2*j+2]
        
        for term in strassen_terms:
            # Fill U (A part)
            for r, c, coeff in term['A']:
                # Global A index: row r, col c (since A1 is cols 0-1)
                u_idx = r * m + c
                U[u_idx][col] += coeff
            
            # Fill V (B part)
            for r, c, coeff in term['B']:
                # Global B index: row r (since B1j is rows 0-1), col 2*j + c
                v_idx = r * p + (2 * j + c)
                V[v_idx][col] += coeff
            
            # Fill W (C part)
            for r, c, coeff in term['C']:
                # Global C index: row r, col 2*j + c
                # W index is k * n + i, where k is col, i is row
                k = 2 * j + c
                i = r
                w_idx = k * n + i
                W[w_idx][col] += coeff
            
            col += 1
        
        # Part 2: A2 * B2j
        # A2 is A[:, 2:4]
        # B2j is B[2:4, 2*j : 2*j+2]
        # Output is C[:, 2*j : 2*j+2]
        
        for term in strassen_terms:
            # Fill U (A part)
            for r, c, coeff in term['A']:
                # Global A index: row r, col 2 + c (since A2 is cols 2-3)
                u_idx = r * m + (2 + c)
                U[u_idx][col] += coeff
            
            # Fill V (B part)
            for r, c, coeff in term['B']:
                # Global B index: row 2 + r (since B2j is rows 2-3), col 2*j + c
                v_idx = (2 + r) * p + (2 * j + c)
                V[v_idx][col] += coeff
            
            # Fill W (C part)
            for r, c, coeff in term['C']:
                # Global C index: row r, col 2*j + c
                k = 2 * j + c
                i = r
                w_idx = k * n + i
                W[w_idx][col] += coeff
            
            col += 1
            
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
