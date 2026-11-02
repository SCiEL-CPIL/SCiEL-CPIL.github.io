import time

DIMENSIONS = [4, 4, 6]

def _add_strassen_term(terms, bi, bk, x_block_col, y_block_row, n, m, p):
    def x_pos(di, dj):
        i = 2*bi + di
        j = 2*x_block_col + dj
        return i * m + j
    
    def y_pos(di, dj):
        j = 2*y_block_row + di
        k = 2*bk + dj
        return j * p + k
    
    def z_pos(di, dj):
        i = 2*bi + di
        k = 2*bk + dj
        return k * n + i
    
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]
    
    for x_c, y_c, z_c in strassen_terms:
        U_dict = {}
        V_dict = {}
        W_dict = {}
        for (di, dj), coeff in x_c.items():
            pos = x_pos(di, dj)
            U_dict[pos] = U_dict.get(pos, 0) + coeff
        for (di, dj), coeff in y_c.items():
            pos = y_pos(di, dj)
            V_dict[pos] = V_dict.get(pos, 0) + coeff
        for (di, dj), coeff in z_c.items():
            pos = z_pos(di, dj)
            W_dict[pos] = W_dict.get(pos, 0) + coeff
        terms.append((U_dict, V_dict, W_dict))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    terms = []
    
    for bi in range(2):
        for bk in range(3):
            _add_strassen_term(terms, bi, bk, 0, 0, n, m, p)
            _add_strassen_term(terms, bi, bk, 1, 1, n, m, p)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t, (U_dict, V_dict, W_dict) in enumerate(terms):
        for pos, coeff in U_dict.items():
            U[pos][t] = coeff
        for pos, coeff in V_dict.items():
            V[pos][t] = coeff
        for pos, coeff in W_dict.items():
            W[pos][t] = coeff
    
    if time.time() > deadline:
        # Should not happen, but just in case
        pass
    
    return {"U": U, "V": V, "W": W}
