"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition: groups of sizes [2, 2, 1] for each dimension
    sizes = [2, 2, 1]
    # Offsets for each group
    offsets = [0, 2, 4]
    
    U = []
    V = []
    W = []
    
    # Strassen 7 terms for 2x2x2
    # Each term: (u_coeffs, v_coeffs, w_coeffs) where each is dict from (local_i, local_j) etc to value
    # For 2x2x2: U index (i,j) in 0..1, V index (j,k) in 0..1, W index (k,i) in 0..1
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        {
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,0): 1, (0,1): 1, (1,1): 1},
        },
        # M2 = (A10+A11)B00
        {
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,0): 1},
            'W': {(0,0): 1, (1,0): 1},
        },
        # M3 = A00(B10-B11)
        {
            'U': {(0,0): 1},
            'V': {(1,0): 1, (1,1): -1},
            'W': {(0,0): 1, (1,0): 1},
        },
        # M4 = A11(B01-B10)
        {
            'U': {(1,1): 1},
            'V': {(0,1): 1, (1,0): -1},
            'W': {(0,0): 1, (1,0): 1},
        },
        # M5 = (A00+A01)B11
        {
            'U': {(0,0): 1, (0,1): 1},
            'V': {(1,1): 1},
            'W': {(0,0): 1, (1,0): 1},
        },
        # M6 = (A10-A00)(B00+B10)
        {
            'U': {(1,0): 1, (0,0): -1},
            'V': {(0,0): 1, (1,0): 1},
            'W': {(0,0): 1, (1,0): 1},
        },
        # M7 = (A01-A11)(B01+B11)
        {
            'U': {(0,1): 1, (1,1): -1},
            'V': {(0,1): 1, (1,1): 1},
            'W': {(0,0): 1, (1,0): 1},
        },
    ]
    
    # Build the decomposition
    columns = []  # list of (U_row_dict, V_row_dict, W_row_dict)
    
    for a in range(3):
        for b in range(3):
            for c in range(3):
                sa, sb, sc = sizes[a], sizes[b], sizes[c]
                i0, j0, k0 = offsets[a], offsets[b], offsets[c]
                
                if sa == 2 and sb == 2 and sc == 2:
                    # Use Strassen
                    for term in strassen_terms:
                        U_dict = {}
                        V_dict = {}
                        W_dict = {}
                        for (di, dj), val in term['U'].items():
                            U_dict[(i0 + di, j0 + dj)] = val
                        for (dj, dk), val in term['V'].items():
                            V_dict[(j0 + dj, k0 + dk)] = val
                        for (dk, di), val in term['W'].items():
                            W_dict[(k0 + dk, i0 + di)] = val
                        columns.append((U_dict, V_dict, W_dict))
                else:
                    # Schoolbook for this block
                    for di in range(sa):
                        for dj in range(sb):
                            for dk in range(sc):
                                U_dict = {(i0 + di, j0 + dj): 1}
                                V_dict = {(j0 + dj, k0 + dk): 1}
                                W_dict = {(k0 + dk, i0 + di): 1}
                                columns.append((U_dict, V_dict, W_dict))
    
    rank = len(columns)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_dict, V_dict, W_dict) in enumerate(columns):
        for (i, j), val in U_dict.items():
            U[i * m + j][col] = val
        for (j, k), val in V_dict.items():
            V[j * p + k][col] = val
        for (k, i), val in W_dict.items():
            W[k * n + i][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
