import time

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
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
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(1, 1)]
        },
        {
            'U': [(0, 1)],
            'V': [(1, 1), (3, -1)],
            'W': [(3, 1)]
        },
        {
            'U': [(3, 1)],
            'V': [(2, 1), (0, -1)],
            'W': [(0, 1), (3, 1)]
        },
        {
            'U': [(0, 1), (1, 1)],
            'V': [(3, 1)],
            'W': [(0, -1)]
        },
        {
            'U': [(2, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(3, 1)]
        },
        {
            'U': [(1, 1), (3, -1)],
            'V': [(2, 1), (3, 1)],
            'W': [(0, 1), (3, 1)]
        }
    ]
    
    sub_problems = [
        {'j_A': 0, 'j_B': 0, 'k_B': 0},
        {'j_A': 0, 'j_B': 0, 'k_B': 2},
        {'j_A': 2, 'j_B': 2, 'k_B': 0},
        {'j_A': 2, 'j_B': 2, 'k_B': 2},
    ]
    
    col = 0
    for sp in sub_problems:
        if time.time() > deadline:
            break
        j_A_off = sp['j_A']
        j_B_off = sp['j_B']
        k_B_off = sp['k_B']
        
        for term in strassen_terms:
            if time.time() > deadline:
                break
            for (local_idx, val) in term['U']:
                i_loc = local_idx // 2
                j_loc = local_idx % 2
                global_i = i_loc
                global_j = j_A_off + j_loc
                U[global_i * m + global_j][col] += val
            
            for (local_idx, val) in term['V']:
                j_loc = local_idx // 2
                k_loc = local_idx % 2
                global_j = j_B_off + j_loc
                global_k = k_B_off + k_loc
                V[global_j * p + global_k][col] += val
            
            for (local_idx, val) in term['W']:
                k_loc = local_idx // 2
                i_loc = local_idx % 2
                global_k = k_B_off + k_loc
                global_i = i_loc
                W[global_k * n + global_i][col] += val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
