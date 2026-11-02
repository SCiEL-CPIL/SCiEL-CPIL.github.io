DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 7
    rank = 7 * num_blocks  # 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms: (u_dict, v_dict, w_dict)
    # Local V indices: 0=B00, 1=B01, 2=B10, 3=B11
    # Local W indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),             # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),             # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),             # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),             # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),             # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),             # M7
    ]
    
    col = 0
    for b in range(num_blocks):
        v_off = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        w_off = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for u_d, v_d, w_d in terms:
            for i, val in u_d.items():
                U[i][col] = val
            for li, val in v_d.items():
                V[v_off[li]][col] = val
            for li, val in w_d.items():
                W[w_off[li]][col] = val
            col += 1
    
    return {"U": U, "V": V, "W": W}
