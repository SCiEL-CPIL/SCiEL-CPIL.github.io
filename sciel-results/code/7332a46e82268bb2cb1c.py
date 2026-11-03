import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # C00 = P1 + P4 - P5 + P7
    # C01 = P2 + P3 - P4 + P5
    # C10 = P6 + P7
    # C11 = P1 - P3 + P5 - P6
    coeff = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 1, -1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [1, 0, -1, 0, 1, -1, 0]
    ]
    
    # Products definitions
    # P1: (A00+A11)(B00+B11)
    # P2: (A10+A11)B00
    # P3: A00(B10-B11)
    # P4: A11(B01-B00)
    # P5: (A00+A10)B11
    # P6: (A10-A00)(B00+B01)
    # P7: (A01-A11)(B10+B11)
    
    products = [
        # (a_indices, b_indices)
        # P1
        ([(0,0), (1,1)], [(0,0), (1,1)]),
        # P2
        ([(1,0), (1,1)], [(0,0)]),
        # P3
        ([(0,0)], [(1,0), (1,1)]), # B10-B11 -> (1,0) +1, (1,1) -1
        # P4
        ([(1,1)], [(0,1), (0,0)]), # B01-B00 -> (0,1) +1, (0,0) -1
        # P5
        ([(0,0), (1,0)], [(1,1)]),
        # P6
        ([(1,0), (0,0)], [(0,0), (0,1)]), # A10-A00 -> (1,0)+1, (0,0)-1; B00+B01 -> (0,0)+1, (0,1)+1
        # P7
        ([(0,1), (1,1)], [(1,0), (1,1)]) # A01-A11 -> (0,1)+1, (1,1)-1; B10+B11 -> (1,0)+1, (1,1)+1
    ]
    
    # Signs for P3, P4, P6, P7
    # P3: A00 * (B10 - B11) -> A: (0,0)+1, B: (1,0)+1, (1,1)-1
    # P4: A11 * (B01 - B00) -> A: (1,1)+1, B: (0,1)+1, (0,0)-1
    # P6: (A10 - A00) * (B00 + B01) -> A: (1,0)+1, (0,0)-1, B: (0,0)+1, (0,1)+1
    # P7: (A01 - A11) * (B10 + B11) -> A: (0,1)+1, (1,1)-1, B: (1,0)+1, (1,1)+1
    
    # Corrected products with signs
    products = [
        # (a_terms, b_terms)
        # P1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # P2
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # P3
        ([(0,0,1)], [(1,0,1), (1,1,-1)]),
        # P4
        ([(1,1,1)], [(0,1,1), (0,0,-1)]),
        # P5
        ([(0,0,1), (1,0,1)], [(1,1,1)]),
        # P6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # P7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)])
    ]
    
    blocks = [
        (0, 0, 0, 0),
        (0, 2, 2, 0),
        (0, 0, 0, 2),
        (0, 2, 2, 2)
    ]
    
    col = 0
    for a_i_off, a_j_off, b_j_off, b_k_off in blocks:
        for p_idx in range(7):
            a_terms, b_terms = products[p_idx]
            for i, j, sign in a_terms:
                U[(i+a_i_off)*4 + (j+a_j_off)][col] += sign
            for j, k, sign in b_terms:
                V[(j+b_j_off)*4 + (k+b_k_off)][col] += sign
            # W coefficients
            # C00: i=0, k=0
            W[(0+b_k_off)*2 + (0+a_i_off)][col] += coeff[0][p_idx]
            # C01: i=0, k=1
            W[(1+b_k_off)*2 + (0+a_i_off)][col] += coeff[1][p_idx]
            # C10: i=1, k=0
            W[(0+b_k_off)*2 + (1+a_i_off)][col] += coeff[2][p_idx]
            # C11: i=1, k=1
            W[(1+b_k_off)*2 + (1+a_i_off)][col] += coeff[3][p_idx]
            col += 1
            
    return {"U": U, "V": V, "W": W}
