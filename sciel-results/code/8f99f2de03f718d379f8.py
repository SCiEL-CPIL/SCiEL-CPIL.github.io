DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (20, rank)
    
    rank = 35
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen 7 products for 2x2 x 2x2
    # For each block b (0..4), columns 2b, 2b+1
    # A indices: i in {0,1}, j in {0,1} -> U index i*m+j
    # B indices: j in {0,1}, k in {0,1} within block -> global k = 2b+k_local
    #   V index: j*p + k_global
    # C indices: i in {0,1}, k_global -> W index: k_global*n + i
    
    # Strassen products: (A_part, B_part)
    # M1: (A00+A11, B00+B11)
    # M2: (A10+A11, B00)
    # M3: (A00, B10-B11)
    # M4: (A11, B01-B10)
    # M5: (A00+A01, B11)
    # M6: (A10-A00, B00+B10)
    # M7: (A01-A11, B01+B11)
    
    # Output formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define each Strassen term as (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict from (i,j) to coefficient
    # b_coeffs: dict from (j,k_local) to coefficient
    # c_coeffs: dict from (i,k_local) to coefficient
    
    terms = []
    
    # M1
    terms.append({
        'a': {(0,0):1, (1,1):1},
        'b': {(0,0):1, (1,1):1},
        'c': {(0,0):1, (1,1):1}
    })
    # M2
    terms.append({
        'a': {(1,0):1, (1,1):1},
        'b': {(0,0):1},
        'c': {(1,0):1, (1,1):-1}
    })
    # M3
    terms.append({
        'a': {(0,0):1},
        'b': {(1,0):1, (1,1):-1},
        'c': {(0,1):1, (1,1):1}
    })
    # M4
    terms.append({
        'a': {(1,1):1},
        'b': {(0,1):1, (1,0):-1},
        'c': {(0,0):1, (1,0):1}
    })
    # M5
    terms.append({
        'a': {(0,0):1, (0,1):1},
        'b': {(1,1):1},
        'c': {(0,0):-1, (0,1):1}
    })
    # M6
    terms.append({
        'a': {(1,0):1, (0,0):-1},
        'b': {(0,0):1, (1,0):1},
        'c': {(1,1):1}
    })
    # M7
    terms.append({
        'a': {(0,1):1, (1,1):-1},
        'b': {(0,1):1, (1,1):1},
        'c': {(0,0):1}
    })
    
    col = 0
    for b in range(5):  # 5 column blocks
        for t in range(7):  # 7 Strassen terms
            term = terms[t]
            for (i,j), coeff in term['a'].items():
                U[i*m+j][col] += coeff
            for (j,k_local), coeff in term['b'].items():
                k_global = 2*b + k_local
                V[j*p + k_global][col] += coeff
            for (i,k_local), coeff in term['c'].items():
                k_global = 2*b + k_local
                W[k_global*n + i][col] += coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
