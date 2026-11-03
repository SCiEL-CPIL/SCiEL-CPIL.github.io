import time

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        {
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (3, 1)],
            'W': [(0, 1), (3, 1)]
        },
        {
            'U': [(0, 1), (3, 1)],
            'V': [(2, 1)],
            'W': [(1, 1), (3, -1)]
        },
        {
            'U': [(0, 1)],
            'V': [(1, 1), (3, -1)],
            'W': [(2, 1), (3, 1)]
        },
        {
            'U': [(3, 1)],
            'V': [(0, 1), (3, 1)],
            'W': [(0, 1), (3, 1)]
        },
        {
            'U': [(1, 1), (3, -1)],
            'V': [(2, 1), (3, 1)],
            'W': [(0, -1), (2, 1)]
        },
        {
            'U': [(0, 1), (2, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(0, 1), (2, 1)]
        },
        {
            'U': [(2, 1), (3, 1)],
            'V': [(0, -1), (3, 1)],
            'W': [(0, -1), (1, 1)]
        }
    ]
    
    products = [
        {'a_col': 0, 'b_row': 0, 'b_col': 0, 'c_row': 0, 'c_col': 0},
        {'a_col': 2, 'b_row': 2, 'b_col': 0, 'c_row': 0, 'c_col': 0},
        {'a_col': 0, 'b_row': 0, 'b_col': 2, 'c_row': 2, 'c_col': 0},
        {'a_col': 2, 'b_row': 2, 'b_col': 2, 'c_row': 2, 'c_col': 0}
    ]
    
    col = 0
    for prod in products:
        for term in strassen_terms:
            if time.time() > deadline:
                break
            for (idx, val) in term['U']:
                u, v = divmod(idx, 2)
                global_i = u
                global_j = v + prod['a_col']
                U[global_i * m + global_j][col] += val
            
            for (idx, val) in term['V']:
                v, w = divmod(idx, 2)
                global_j = v + prod['b_row']
                global_k = w + prod['b_col']
                V[global_j * p + global_k][col] += val
            
            for (idx, val) in term['W']:
                w, u = divmod(idx, 2)
                global_k = w + prod['c_row']
                global_i = u + prod['c_col']
                W[global_k * n + global_i][col] += val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
