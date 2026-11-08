import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # Strassen 2x2 U, V, W matrices (4x7 each)
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 0, 1],
        [0, 0, -1, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    outer_terms = [
        {'A_blocks': [(0, 1), (12, 1)], 'B_blocks': [(0, 1), (12, 1)],
         'contrib': {(0,0):[(0,0,1),(2,2,1)], (0,1):[(0,1,1),(2,3,1)],
                     (1,0):[(1,0,1),(3,2,1)], (1,1):[(1,1,1),(3,3,1)]}},
        {'A_blocks': [(8, 1), (12, 1)], 'B_blocks': [(0, 1)],
         'contrib': {(0,0):[(2,0,1),(2,2,-1)], (0,1):[(2,1,1),(2,3,-1)],
                     (1,0):[(3,0,1),(3,2,-1)], (1,1):[(3,1,1),(3,3,-1)]}},
        {'A_blocks': [(0, 1)], 'B_blocks': [(4, 1), (12, -1)],
         'contrib': {(0,0):[(0,2,1),(2,2,1)], (0,1):[(0,3,1),(2,3,1)],
                     (1,0):[(1,2,1),(3,2,1)], (1,1):[(1,3,1),(3,3,1)]}},
        {'A_blocks': [(12, 1)], 'B_blocks': [(8, 1), (0, -1)],
         'contrib': {(0,0):[(0,0,1),(2,0,1)], (0,1):[(0,1,1),(2,1,1)],
                     (1,0):[(1,0,1),(3,0,1)], (1,1):[(1,1,1),(3,1,1)]}},
        {'A_blocks': [(0, 1), (4, 1)], 'B_blocks': [(12, 1)],
         'contrib': {(0,0):[(0,2,1),(0,0,-1)], (0,1):[(0,3,1),(0,1,-1)],
                     (1,0):[(1,2,1),(1,0,-1)], (1,1):[(1,3,1),(1,1,-1)]}},
        {'A_blocks': [(8, 1), (0, -1)], 'B_blocks': [(0, 1), (8, 1)],
         'contrib': {(0,0):[(2,2,1)], (0,1):[(2,3,1)],
                     (1,0):[(3,2,1)], (1,1):[(3,3,1)]}},
        {'A_blocks': [(4, 1), (12, -1)], 'B_blocks': [(4, 1), (12, 1)],
         'contrib': {(0,0):[(0,0,1)], (0,1):[(0,1,1)],
                     (1,0):[(1,0,1)], (1,1):[(1,1,1)]}}
    ]
    
    for t, term in enumerate(outer_terms):
        if time.time() > deadline:
            break
        for s in range(7):
            col = t * 7 + s
            
            for block_off, coeff in term['A_blocks']:
                for local_row in range(4):
                    global_row = block_off + local_row
                    U[global_row][col] += coeff * U2[local_row][s]
            
            for block_off, coeff in term['B_blocks']:
                for local_row in range(4):
                    global_row = block_off + local_row
                    V[global_row][col] += coeff * V2[local_row][s]
            
            for lr in range(2):
                for lc in range(2):
                    local_w_row = lr * 2 + lc
                    for (c_row, c_col, w_coeff) in term['contrib'][(lr, lc)]:
                        global_w_row = c_col * 4 + c_row
                        W[global_w_row][col] += w_coeff * W2[local_w_row][s]
    
    return {"U": U, "V": V, "W": W}
