"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # We'll build rank-1 terms
    terms = []  # Each term is (U_dict, V_dict, W_dict)
    
    # Strassen's 7 terms for the 2x2x2 part (k=0,1)
    # M1 = (A00+A10)(B00+B11)
    terms.append({
        'U': {0: 1, 2: 1},  # A[0,0], A[1,0]
        'V': {0: 1, 6: 1},  # B[0,0], B[1,1]
        'W': {0: 1, 1: -1, 2: 0, 3: 1}  # C[0,0], C[1,0], C[0,1], C[1,1]
    })
    # M2 = (A00+A10)B01
    terms.append({
        'U': {0: 1, 2: 1},
        'V': {1: 1},  # B[0,1]
        'W': {0: 0, 1: 1, 2: 0, 3: -1}  # C[1,0] gets +M2, C[1,1] gets -M2
    })
    # M3 = A00(B01-B10)
    terms.append({
        'U': {0: 1},
        'V': {1: 1, 5: -1},  # B[0,1], B[1,0]
        'W': {0: 0, 1: 0, 2: 1, 3: -1}  # C[0,1] gets +M3, C[1,1] gets -M3
    })
    # M4 = A10(B00-B10)
    terms.append({
        'U': {2: 1},
        'V': {0: 1, 5: -1},
        'W': {0: -1, 1: 0, 2: 1, 3: 0}  # C[0,0] gets -M4, C[0,1] gets +M4
    })
    # M5 = (A01+A10)B00
    terms.append({
        'U': {1: 1, 2: 1},  # A[0,1], A[1,0]
        'V': {0: 1},
        'W': {0: 1, 1: 1, 2: 0, 3: 0}  # C[0,0] gets +M5, C[1,0] gets +M5
    })
    # M6 = (A01+A10)B11
    terms.append({
        'U': {1: 1, 2: 1},
        'V': {6: 1},
        'W': {0: 0, 1: 0, 2: 1, 3: -1}  # C[0,1] gets +M6, C[1,1] gets -M6
    })
    # M7 = (A00-A10)(B01+B11)
    terms.append({
        'U': {0: 1, 2: -1},
        'V': {1: 1, 6: 1},
        'W': {0: -1, 1: 1, 2: 0, 3: 0}  # C[0,0] gets -M7, C[1,0] gets +M7
    })
    
    # Schoolbook terms for k=2,3,4 (4 terms per column)
    for k in range(2, p):
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                terms.append({
                    'U': {u_idx: 1},
                    'V': {v_idx: 1},
                    'W': {w_idx: 1}
                })
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, term in enumerate(terms):
        for idx, val in term['U'].items():
            U[idx][col] = val
        for idx, val in term['V'].items():
            V[idx][col] = val
        for idx, val in term['W'].items():
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
