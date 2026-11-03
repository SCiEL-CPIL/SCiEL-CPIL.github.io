import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use block decomposition: 2x2 blocks of 3x3
    # Outer rank 7 (Strassen), Inner rank 27 (Schoolbook)
    # Total rank 189
    
    # Strassen coefficients for 2x2
    # U_a, V_b are 2x2 matrices of coefficients
    # M_k = U_k * V_k
    # C = sum M_k
    
    # Strassen terms (U, V)
    # U is 2x2, V is 2x2
    # Indices 0,1 for blocks
    strassen_U = [
        [[1, 0], [1, 1]], # M1: A00+A11 (wait, U is applied to A)
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11) -> U1 = [[1,0],[1,1]]? No.
        # A is [[A00, A01], [A10, A11]]
        # U1 should be such that U1 * A = A00+A11?
        # No, U is a matrix of coefficients for the blocks.
        # Let's define the linear maps.
        
        # Term 1: (A00 + A11) * (B00 + B11)
        # A_part: A00 + A11. Coeffs: A00:1, A11:1.
        # B_part: B00 + B11. Coeffs: B00:1, B11:1.
    ]
    
    # It is easier to define the 7 terms explicitly.
    # Each term is a pair of linear combinations of the 4 blocks of A and B.
    # Blocks are indexed 00, 01, 10, 11.
    
    terms = [
        # (A_coeffs, B_coeffs)
        # A_coeffs is dict {block_idx: coeff}
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),             # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),            # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),            # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),             # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),  # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),  # M7
    ]
    
    # Schoolbook 3x3 terms
    # 27 terms. Each is (i, j, k) for A[i,k] * B[k,j] -> C[i,j]
    schoolbook_terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                schoolbook_terms.append((i, j, k))
                
    rank = 7 * 27
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    col = 0
    for t_idx in range(7):
        a_coeffs, b_coeffs = terms[t_idx]
        for sb_idx in range(27):
            i, j, k = schoolbook_terms[sb_idx]
            # This term computes A_eff[i, k] * B_eff[k, j]
            
            # Update U (A side)
            # A_eff is a linear combo of blocks of A.
            # A_eff[i, k] = sum_{block} coeff * A_block[i, k]
            for (ba, bb), coeff in a_coeffs.items():
                # Block (ba, bb) is at rows ba*3..ba*3+2, cols bb*3..bb*3+2
                # Element (i, k) in the block is at global row ba*3+i, col bb*3+k
                # Flatten index (row-major): row * 6 + col
                r = (ba * 3 + i) * 6 + (bb * 3 + k)
                U[r][col] += coeff
                
            # Update V (B side)
            for (ba, bb), coeff in b_coeffs.items():
                # B_eff[k, j]
                # Element (k, j) in the block is at global row ba*3+k, col bb*3+j
                # Flatten index (row-major): row * 6 + col
                r = (ba * 3 + k) * 6 + (bb * 3 + j)
                V[r][col] += coeff
                
            # Update W (C side)
            # The product contributes to C_eff[i, j]
            # C_eff is added to C blocks.
            # Which C blocks are affected?
            # Strassen output C is sum of M_k * C_k
            # C_k is a 2x2 matrix of 3x3 blocks.
            # We need to know which C blocks each M_k contributes to.
            
            # Strassen C reconstruction:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # We need to map t_idx (0..6) to C block contributions.
            # t=0 (M1): C00:1, C11:1
            # t=1 (M2): C10:1, C11:-1
            # t=2 (M3): C01:1, C11:1
            # t=3 (M4): C00:1, C10:1
            # t=4 (M5): C00:-1, C01:1
            # t=5 (M6): C11:1
            # t=6 (M7): C00:1
            
            c_contribs = {}
            if t_idx == 0: c_contribs = {(0,0): 1, (1,1): 1}
            elif t_idx == 1: c_contribs = {(1,0): 1, (1,1): -1}
            elif t_idx == 2: c_contribs = {(0,1): 1, (1,1): 1}
            elif t_idx == 3: c_contribs = {(0,0): 1, (1,0): 1}
            elif t_idx == 4: c_contribs = {(0,0): -1, (0,1): 1}
            elif t_idx == 5: c_contribs = {(1,1): 1}
            elif t_idx == 6: c_contribs = {(0,0): 1}
            
            for (ca, cb), coeff in c_contribs.items():
                # C block (ca, cb) element (i, j)
                # Global Row I = ca*3+i, Global Col J = cb*3+j
                # Convention W[k*n+i] implies Index = J * n + I (Column Major)
                r = (cb * 3 + j) * 6 + (ca * 3 + i)
                W[r][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
