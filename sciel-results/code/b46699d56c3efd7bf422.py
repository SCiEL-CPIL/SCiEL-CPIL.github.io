import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Use 2x2 blocks of 3x3
    # Strassen for 2x2 (Rank 7)
    # Schoolbook for 3x3 (Rank 27)
    # Total Rank 189
    
    terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
    ]
    
    schoolbook_terms = [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]
    
    rank = 7 * 27
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    col = 0
    for t_idx in range(7):
        a_coeffs, b_coeffs = terms[t_idx]
        for sb_idx in range(27):
            i, j, k = schoolbook_terms[sb_idx]
            
            for (ba, bb), coeff in a_coeffs.items():
                r = (ba * 3 + i) * 6 + (bb * 3 + k)
                U[r, col] += coeff
                
            for (ba, bb), coeff in b_coeffs.items():
                r = (ba * 3 + k) * 6 + (bb * 3 + j)
                V[r, col] += coeff
                
            c_contribs = {}
            if t_idx == 0: c_contribs = {(0,0): 1, (1,1): 1}
            elif t_idx == 1: c_contribs = {(1,0): 1, (1,1): -1}
            elif t_idx == 2: c_contribs = {(0,1): 1, (1,1): 1}
            elif t_idx == 3: c_contribs = {(0,0): 1, (1,0): 1}
            elif t_idx == 4: c_contribs = {(0,0): -1, (0,1): 1}
            elif t_idx == 5: c_contribs = {(1,1): 1}
            elif t_idx == 6: c_contribs = {(0,0): 1}
            
            for (ca, cb), coeff in c_contribs.items():
                r = (cb * 3 + j) * 6 + (ca * 3 + i)
                W[r, col] += coeff
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
