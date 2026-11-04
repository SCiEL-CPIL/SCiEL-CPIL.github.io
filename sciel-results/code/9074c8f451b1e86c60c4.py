import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    strassen_terms = [
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1)]},
        {'U': [(0, 1)], 'V': [(2, 1), (3, -1)], 'W': [(1, 1), (3, -1)]},
        {'U': [(3, 1)], 'V': [(1, 1), (3, 1)], 'W': [(0, 1), (2, 1)]},
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (1, 1)]},
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1), (3, -1)]}
    ]

    def add_block(x_off_i, x_off_j, y_off_j, y_off_k, z_off_i, z_off_k):
        nonlocal col
        if time.time() > deadline:
            return
        for term in strassen_terms:
            for idx, coeff in term['U']:
                li = idx // 2
                lj = idx % 2
                global_i = x_off_i + li
                global_j = x_off_j + lj
                u_idx = global_i * m + global_j
                U[u_idx][col] += coeff
            
            for idx, coeff in term['V']:
                li = idx // 2
                lj = idx % 2
                global_j = y_off_j + li
                global_k = y_off_k + lj
                v_idx = global_j * p + global_k
                V[v_idx][col] += coeff
            
            for idx, coeff in term['W']:
                li = idx // 2
                lj = idx % 2
                global_k = z_off_k + li
                global_i = z_off_i + lj
                w_idx = global_k * n + global_i
                W[w_idx][col] += coeff
            
            col += 1

    add_block(0, 0, 0, 0, 0, 0)
    add_block(0, 0, 0, 2, 0, 2)
    add_block(0, 0, 0, 4, 0, 4)
    add_block(0, 2, 2, 0, 0, 0)
    add_block(0, 2, 2, 2, 0, 2)
    add_block(0, 2, 2, 4, 0, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
