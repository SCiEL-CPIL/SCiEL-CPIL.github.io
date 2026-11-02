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
    
    # Strassen local coefficients
    u_local = [
        [1, 0, 1, 0, 1, -1, 0], # x00
        [0, 0, 0, 0, 1, 0, 1],  # x01
        [0, 1, 0, 0, 0, 1, 0],  # x10
        [1, 1, 0, 1, 0, 0, -1]  # x11
    ]
    
    v_local = [
        [1, 1, 0, -1, 0, 1, 0], # y00
        [0, 0, 1, 0, 0, 1, 0],  # y01
        [0, 0, 0, 1, 0, 0, 1],  # y10
        [1, 0, -1, 0, 1, 0, 1]  # y11
    ]
    
    w_local = [
        [1, 0, 0, 1, -1, 0, 1], # z00
        [0, 0, 1, 0, 1, 0, 0],  # z01
        [0, 1, 0, 1, 0, 0, 0],  # z10
        [1, -1, 1, 0, 0, 1, 0]  # z11
    ]
    
    blocks = [
        ([0, 1], [0, 1, 4, 5], [0, 1, 2, 3]),
        ([0, 1], [2, 3, 6, 7], [4, 5, 6, 7]),
        ([2, 3], [8, 9, 12, 13], [0, 1, 2, 3]),
        ([2, 3], [10, 11, 14, 15], [4, 5, 6, 7]),
    ]
    
    for b_idx, (u_rows, v_rows, w_rows) in enumerate(blocks):
        if time.time() > deadline:
            break
        for k in range(7):
            col = b_idx * 7 + k
            for i in range(4):
                U[u_rows[i]][col] += u_local[i][k]
                V[v_rows[i]][col] += v_local[i][k]
                W[w_rows[i]][col] += w_local[i][k]
                
    return {"U": U, "V": V, "W": W}

