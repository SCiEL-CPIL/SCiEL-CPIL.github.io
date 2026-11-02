import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen factors for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # u vectors for A
    u_strassen = [
        [1, 0, 0, 1],  # a + d
        [0, 0, 1, 1],  # c + d
        [1, 0, 0, 0],  # a
        [0, 0, 0, 1],  # d
        [1, 1, 0, 0],  # a + b
        [-1, 0, 1, 0], # c - a
        [0, 1, 0, -1]  # b - d
    ]
    
    # v vectors for B
    v_strassen = [
        [1, 0, 0, 1],  # e + h
        [1, 0, 0, 0],  # e
        [0, 1, 0, -1], # f - h
        [0, 0, 1, 0],  # g
        [0, 0, 0, 1],  # h
        [1, 1, 0, 0],  # e + f
        [0, 0, 1, 1]   # g + h
    ]
    
    # w vectors for P
    # P = [[m1+m4-m5+m7, m5+m6], [m2+m3, m1-m2+m3+m4]]
    # Indices: 0=P00, 1=P01, 2=P10, 3=P11
    w_strassen = [
        [1, 0, 0, 1],  # m1
        [0, 0, 1, -1], # m2
        [0, 0, 1, 1],  # m3
        [1, 0, 0, 1],  # m4
        [-1, 1, 0, 0], # m5
        [0, 1, 0, 0],  # m6
        [1, 0, 0, 0]   # m7
    ]
    
    # Mapping of block indices to global indices
    # A is 2x4. A_L is cols 0,1. A_R is cols 2,3.
    # B is 4x4. 
    # B_TL is rows 0,1 cols 0,1.
    # B_TR is rows 0,1 cols 2,3.
    # B_BL is rows 2,3 cols 0,1.
    # B_BR is rows 2,3 cols 2,3.
    
    # C is 2x4. C_L is cols 0,1. C_R is cols 2,3.
    # W index is k*n + i.
    # C_L: k in 0,1. C_R: k in 2,3.
    
    def add_block_product(u_offset, v_offset, w_offset, block_type):
        # block_type: 0: A_L B_TL (C_L), 1: A_L B_TR (C_R), 2: A_R B_BL (C_L), 3: A_R B_BR (C_R)
        for s in range(7):
            col = 7 * block_type + s
            
            # Fill U
            # u_strassen[s] is [a, b, c, d] coeffs
            # A_L: i in 0,1, j in 0,1. Global index i*4 + j.
            # A_R: i in 0,1, j in 2,3. Global index i*4 + j.
            if block_type in [0, 1]: # A_L
                for i in range(2):
                    for j in range(2):
                        # u_strassen[s] is [a, b, c, d] -> [A00, A01, A10, A11]
                        # j=0 is col 0, j=1 is col 1
                        # i=0 is row 0, i=1 is row 1
                        # u_strassen[s][0] is a (A00), [1] is b (A01), [2] is c (A10), [3] is d (A11)
                        if j == 0:
                            if i == 0:
                                U[i * 4 + j, col] += u_strassen[s][0]
                            else:
                                U[i * 4 + j, col] += u_strassen[s][2]
                        else:
                            if i == 0:
                                U[i * 4 + j, col] += u_strassen[s][1]
                            else:
                                U[i * 4 + j, col] += u_strassen[s][3]
            else: # A_R
                for i in range(2):
                    for j in range(2):
                        # j_global = j + 2
                        if j == 0:
                            if i == 0:
                                U[i * 4 + (j+2), col] += u_strassen[s][0]
                            else:
                                U[i * 4 + (j+2), col] += u_strassen[s][2]
                        else:
                            if i == 0:
                                U[i * 4 + (j+2), col] += u_strassen[s][1]
                            else:
                                U[i * 4 + (j+2), col] += u_strassen[s][3]
                                
            # Fill V
            # v_strassen[s] is [e, f, g, h] -> [B00, B01, B10, B11]
            if block_type == 0: # B_TL (rows 0,1 cols 0,1)
                for r in range(2):
                    for c in range(2):
                        if r == 0:
                            if c == 0:
                                V[r * 4 + c, col] += v_strassen[s][0]
                            else:
                                V[r * 4 + c, col] += v_strassen[s][1]
                        else:
                            if c == 0:
                                V[r * 4 + c, col] += v_strassen[s][2]
                            else:
                                V[r * 4 + c, col] += v_strassen[s][3]
            elif block_type == 1: # B_TR (rows 0,1 cols 2,3)
                for r in range(2):
                    for c in range(2):
                        # c_global = c + 2
                        if r == 0:
                            if c == 0:
                                V[r * 4 + (c+2), col] += v_strassen[s][0]
                            else:
                                V[r * 4 + (c+2), col] += v_strassen[s][1]
                        else:
                            if c == 0:
                                V[r * 4 + (c+2), col] += v_strassen[s][2]
                            else:
                                V[r * 4 + (c+2), col] += v_strassen[s][3]
            elif block_type == 2: # B_BL (rows 2,3 cols 0,1)
                for r in range(2):
                    for c in range(2):
                        # r_global = r + 2
                        if r == 0:
                            if c == 0:
                                V[(r+2) * 4 + c, col] += v_strassen[s][0]
                            else:
                                V[(r+2) * 4 + c, col] += v_strassen[s][1]
                        else:
                            if c == 0:
                                V[(r+2) * 4 + c, col] += v_strassen[s][2]
                            else:
                                V[(r+2) * 4 + c, col] += v_strassen[s][3]
            else: # B_BR (rows 2,3 cols 2,3)
                for r in range(2):
                    for c in range(2):
                        # r_global = r + 2, c_global = c + 2
                        if r == 0:
                            if c == 0:
                                V[(r+2) * 4 + (c+2), col] += v_strassen[s][0]
                            else:
                                V[(r+2) * 4 + (c+2), col] += v_strassen[s][1]
                        else:
                            if c == 0:
                                V[(r+2) * 4 + (c+2), col] += v_strassen[s][2]
                            else:
                                V[(r+2) * 4 + (c+2), col] += v_strassen[s][3]
                                
            # Fill W
            # w_strassen[s] is [P00, P01, P10, P11]
            if block_type in [0, 2]: # C_L (k in 0,1)
                for i in range(2):
                    for k in range(2):
                        # W index k*n + i = k*2 + i
                        # P index: 0=P00(i=0,k=0), 1=P01(i=0,k=1), 2=P10(i=1,k=0), 3=P11(i=1,k=1)
                        # Wait, P is 2x2. P[i][k].
                        # P00: i=0, k=0.
                        # P01: i=0, k=1.
                        # P10: i=1, k=0.
                        # P11: i=1, k=1.
                        p_idx = i * 2 + k
                        W[k * 2 + i, col] += w_strassen[s][p_idx]
            else: # C_R (k in 2,3)
                for i in range(2):
                    for k in range(2):
                        # k_global = k + 2
                        p_idx = i * 2 + k
                        W[(k+2) * 2 + i, col] += w_strassen[s][p_idx]

    add_block_product(0, 0, 0, 0)
    add_block_product(0, 1, 1, 1)
    add_block_product(2, 2, 2, 2)
    add_block_product(2, 3, 3, 3)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
