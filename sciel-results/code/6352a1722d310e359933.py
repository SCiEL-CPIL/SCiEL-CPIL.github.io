import time

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}, 'C': {0: 1, 3: 1}},
        {'A': {2: 1, 3: 1}, 'B': {0: 1}, 'C': {2: 1, 3: -1}},
        {'A': {0: 1}, 'B': {2: 1, 3: -1}, 'C': {1: 1, 3: 1}},
        {'A': {3: 1}, 'B': {1: 1, 0: -1}, 'C': {0: 1, 2: 1}},
        {'A': {0: 1, 1: 1}, 'B': {3: 1}, 'C': {0: -1, 1: 1}},
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}, 'C': {3: 1}},
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}, 'C': {0: 1}}
    ]
    
    for b in range(8):
        if time.time() > deadline:
            break
        j_idx = b // 4
        k_idx = b % 4
        j_base = 2 * j_idx
        k_base = 2 * k_idx
        
        for t in range(7):
            c = 7 * b + t
            term = strassen_terms[t]
            
            for loc_idx, coeff in term['A'].items():
                r = loc_idx // 2
                c_loc = loc_idx % 2
                i = r
                j = j_base + c_loc
                u_idx = i * m + j
                U[u_idx][c] += coeff
            
            for loc_idx, coeff in term['B'].items():
                r = loc_idx // 2
                c_loc = loc_idx % 2
                j = j_base + r
                k = k_base + c_loc
                v_idx = j * p + k
                V[v_idx][c] += coeff
            
            for loc_idx, coeff in term['C'].items():
                r = loc_idx // 2
                c_loc = loc_idx % 2
                i = r
                k = k_base + c_loc
                w_idx = k * n + i
                W[w_idx][c] += coeff
                
    return {"U": U, "V": V, "W": W}
