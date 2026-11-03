import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_j_offset, b_j_offset, k_start, k_end, col_offset):
        nonlocal col
        if time.time() > deadline:
            return col_offset
            
        strassen_terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
        ]
        
        for l in range(7):
            u_dict, v_dict, w_dict = strassen_terms[l]
            r = col_offset + l
            
            for (li, lj), val in u_dict.items():
                gj = a_j_offset + lj
                idx_u = li * m + gj
                U[idx_u][r] += val
                
            for (lj, lk), val in v_dict.items():
                gj = b_j_offset + lj
                gk = k_start + lk
                idx_v = gj * p + gk
                V[idx_v][r] += val
                
            for (lk, li), val in w_dict.items():
                gk = k_start + lk
                idx_w = gk * n + li
                W[idx_w][r] += val
                
        return col_offset + 7

    def add_schoolbook(a_j_offset, b_j_offset, k_idx, col_offset):
        nonlocal col
        if time.time() > deadline:
            return col_offset

        terms = [
            ((0,0), (0,0), (0,0)),
            ((0,1), (1,0), (0,0)),
            ((1,0), (0,0), (1,0)),
            ((1,1), (1,0), (1,0))
        ]
        
        for l in range(4):
            (li, lj), (vj, vk), (wk, wi) = terms[l]
            r = col_offset + l
            
            gj = a_j_offset + lj
            idx_u = li * m + gj
            U[idx_u][r] += 1
            
            gj_v = b_j_offset + vj
            gk = k_idx + vk
            idx_v = gj_v * p + gk
            V[idx_v][r] += 1
            
            gk_w = k_idx + wk
            idx_w = gk_w * n + wi
            W[idx_w][r] += 1
            
        return col_offset + 4

    col = add_strassen(0, 0, 0, 2, col)
    col = add_strassen(0, 0, 2, 4, col)
    col = add_strassen(0, 0, 4, 6, col)
    col = add_schoolbook(0, 0, 6, col)
    
    col = add_strassen(2, 2, 0, 2, col)
    col = add_strassen(2, 2, 2, 4, col)
    col = add_strassen(2, 2, 4, 6, col)
    col = add_schoolbook(2, 2, 6, col)
    
    return {"U": U, "V": V, "W": W}
