import time

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 4
    rank = 28
    
    # Local Strassen factors for 2x2x2
    # U_local: 4x7, V_local: 4x7, W_local: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # M1
    U1 = {0: 1, 3: 1}
    V1 = {0: 1, 3: 1}
    W1 = {0: 1, 3: 1}
    
    # M2
    U2 = {2: 1, 3: 1}
    V2 = {0: 1}
    W2 = {1: 1, 3: -1}
    
    # M3
    U3 = {0: 1}
    V3 = {1: 1, 3: -1}
    W3 = {2: 1, 3: 1}
    
    # M4
    U4 = {3: 1}
    V4 = {2: 1, 0: -1}
    W4 = {0: 1, 1: 1}
    
    # M5
    U5 = {0: 1, 1: 1}
    V5 = {3: 1}
    W5 = {0: -1, 2: 1}
    
    # M6
    U6 = {2: 1, 0: -1}
    V6 = {0: 1, 1: 1}
    W6 = {3: 1}
    
    # M7
    U7 = {1: 1, 3: -1}
    V7 = {2: 1, 3: 1}
    W7 = {0: 1}
    
    strassen_terms = [
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6),
        (U7, V7, W7)
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Blocks:
    # 0: j in [0,1], k in [0,1]
    # 1: j in [0,1], k in [2,3]
    # 2: j in [2,3], k in [0,1]
    # 3: j in [2,3], k in [2,3]
    
    for block in range(4):
        j_offset = (block // 2) * 2
        k_offset = (block % 2) * 2
        
        for U_loc, V_loc, W_loc in strassen_terms:
            for u_idx, val in U_loc.items():
                i = u_idx // 2
                j_loc = u_idx % 2
                j = j_offset + j_loc
                global_u_idx = i * m + j
                U[global_u_idx][col] += val
            
            for v_idx, val in V_loc.items():
                j_loc = v_idx // 2
                k_loc = v_idx % 2
                j = j_offset + j_loc
                k = k_offset + k_loc
                global_v_idx = j * p + k
                V[global_v_idx][col] += val
            
            for w_idx, val in W_loc.items():
                k_loc = w_idx // 2
                i = w_idx % 2
                k = k_offset + k_loc
                global_w_idx = k * n + i
                W[global_w_idx][col] += val
            
            col += 1
            
            if time.time() - start_time > 100:
                break
        if time.time() - start_time > 100:
            break
            
    return {"U": U, "V": V, "W": W}
