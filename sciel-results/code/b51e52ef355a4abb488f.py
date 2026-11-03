import time

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)] # 8 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 8 x 28
    
    # Define Strassen factors for 2x2x2
    # Each factor is a tuple (u_vec, v_vec, w_vec)
    # u_vec corresponds to X[0,0], X[0,1], X[1,0], X[1,1]
    # v_vec corresponds to Y[0,0], Y[0,1], Y[1,0], Y[1,1]
    # w_vec corresponds to Z[0,0], Z[0,1], Z[1,0], Z[1,1]
    
    strassen_factors = [
        # m1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # m2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # m3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # m4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # m5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]),
        # m6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # m7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Define index mappings for each sub-problem
    # Sub-problem 1: A_top, B_top_left
    # X (A) indices: 0, 1, 4, 5
    # Y (B) indices: 0, 1, 4, 5
    # Z (C) indices: 0, 2, 1, 3 (mapping z00->0, z01->2, z10->1, z11->3)
    
    sp1 = {
        'x': [0, 1, 4, 5],
        'y': [0, 1, 4, 5],
        'z': [0, 2, 1, 3]
    }
    
    # Sub-problem 2: A_top, B_top_right
    # X (A) indices: 0, 1, 4, 5
    # Y (B) indices: 2, 3, 6, 7
    # Z (C) indices: 4, 6, 5, 7 (mapping z00->4, z01->6, z10->5, z11->7)
    
    sp2 = {
        'x': [0, 1, 4, 5],
        'y': [2, 3, 6, 7],
        'z': [4, 6, 5, 7]
    }
    
    # Sub-problem 3: A_bot, B_bot_left
    # X (A) indices: 2, 3, 6, 7
    # Y (B) indices: 8, 9, 12, 13
    # Z (C) indices: 0, 2, 1, 3
    
    sp3 = {
        'x': [2, 3, 6, 7],
        'y': [8, 9, 12, 13],
        'z': [0, 2, 1, 3]
    }
    
    # Sub-problem 4: A_bot, B_bot_right
    # X (A) indices: 2, 3, 6, 7
    # Y (B) indices: 10, 11, 14, 15
    # Z (C) indices: 4, 6, 5, 7
    
    sp4 = {
        'x': [2, 3, 6, 7],
        'y': [10, 11, 14, 15],
        'z': [4, 6, 5, 7]
    }
    
    sub_problems = [sp1, sp2, sp3, sp4]
    
    col = 0
    for sp in sub_problems:
        for u_vec, v_vec, w_vec in strassen_factors:
            # Fill U
            for idx, val in enumerate(u_vec):
                if val != 0:
                    U[sp['x'][idx]][col] = val
            
            # Fill V
            for idx, val in enumerate(v_vec):
                if val != 0:
                    V[sp['y'][idx]][col] = val
            
            # Fill W
            for idx, val in enumerate(w_vec):
                if val != 0:
                    W[sp['z'][idx]][col] = val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
