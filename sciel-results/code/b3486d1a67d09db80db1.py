"""Matrix multiplication 3x4x4 rank-minimal decomposition."""
DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = 3, 4, 4
    terms = []
    
    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        # Term 1: (A00+A11)(B00+B11) -> C00 + C11
        t = {'U': {(i0*4+j0): 1, (i1*4+j1): 1},
             'V': {(j0*4+k0): 1, (j1*4+k1): 1},
             'W': {(k0*3+i0): 1, (k1*3+i1): 1}}
        terms.append(t)
        # Term 2: (A10+A11)B00 -> C10
        t = {'U': {(i1*4+j0): 1, (i1*4+j1): 1},
             'V': {(j0*4+k0): 1},
             'W': {(k0*3+i1): 1}}
        terms.append(t)
        # Term 3: A00(B10-B11) -> C01
        t = {'U': {(i0*4+j0): 1},
             'V': {(j1*4+k0): 1, (j1*4+k1): -1},
             'W': {(k1*3+i0): 1}}
        terms.append(t)
        # Term 4: A11(B01-B00) -> C00 + C10
        t = {'U': {(i1*4+j1): 1},
             'V': {(j0*4+k1): 1, (j0*4+k0): -1},
             'W': {(k0*3+i0): 1, (k0*3+i1): 1}}
        terms.append(t)
        # Term 5: (A00+A10)B11 -> C01 - C00
        t = {'U': {(i0*4+j0): 1, (i1*4+j0): 1},
             'V': {(j1*4+k1): 1},
             'W': {(k1*3+i0): 1, (k0*3+i0): -1}}
        terms.append(t)
        # Term 6: (A11-A10)(B00+B10) -> C11
        t = {'U': {(i1*4+j1): 1, (i1*4+j0): -1},
             'V': {(j0*4+k0): 1, (j1*4+k0): 1},
             'W': {(k1*3+i1): 1}}
        terms.append(t)
        # Term 7: (A01-A11)(B01+B11) -> C00
        t = {'U': {(i0*4+j1): 1, (i1*4+j1): -1},
             'V': {(j0*4+k1): 1, (j1*4+k1): 1},
             'W': {(k0*3+i0): 1}}
        terms.append(t)
    
    def add_trivial_block(i, j0, j1, k0, k1):
        for j in [j0, j1]:
            for k in [k0, k1]:
                t = {'U': {(i*4+j): 1},
                     'V': {(j*4+k): 1},
                     'W': {(k*3+i): 1}}
                terms.append(t)
    
    # Strassen blocks for rows {0,1}
    add_strassen_block(0, 1, 0, 1, 0, 1)
    add_strassen_block(0, 1, 2, 3, 0, 1)
    add_strassen_block(0, 1, 0, 1, 2, 3)
    add_strassen_block(0, 1, 2, 3, 2, 3)
    
    # Trivial blocks for row {2}
    add_trivial_block(2, 0, 1, 0, 1)
    add_trivial_block(2, 2, 3, 0, 1)
    add_trivial_block(2, 0, 1, 2, 3)
    add_trivial_block(2, 2, 3, 2, 3)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(12)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(12)]
    
    for col, t in enumerate(terms):
        for mat, data in t.items():
            for row, coeff in data.items():
                if mat == 'U':
                    U[row][col] = coeff
                elif mat == 'V':
                    V[row][col] = coeff
                elif mat == 'W':
                    W[row][col] = coeff
    
    return {"U": U, "V": V, "W": W}
