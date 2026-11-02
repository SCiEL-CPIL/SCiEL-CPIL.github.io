import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    deadline = time.time() + 140
    
    rank = 56
    nm = n * m
    mp = m * p
    pn = p * n
    
    U_mat = [[0] * rank for _ in range(nm)]
    V_mat = [[0] * rank for _ in range(mp)]
    W_mat = [[0] * rank for _ in range(pn)]
    
    # Strassen coefficients for 2x2x2
    # u, v are size 4 (flattened 2x2)
    # w is size 4 (flattened 2x2)
    strassen_u = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, 1], # m2
        [1, 0, 0, 0], # m3
        [0, 0, 0, 1], # m4
        [1, 1, 0, 0], # m5
        [-1, 0, 1, 0],# m6
        [0, 1, 0, -1] # m7
    ]
    strassen_v = [
        [1, 0, 0, 1], # m1
        [1, 0, 0, 0], # m2
        [0, 1, 0, -1],# m3
        [-1, 0, 1, 0],# m4
        [0, 0, 0, 1], # m5
        [1, 1, 0, 0], # m6
        [0, 0, 1, 1]  # m7
    ]
    strassen_w = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, -1],# m2
        [0, 1, 0, 1], # m3
        [1, 0, 1, 0], # m4
        [-1, 1, 0, 0],# m5
        [0, 0, 0, 1], # m6
        [1, 0, 0, 0]  # m7
    ]
    
    # 4 blocks of B (and C)
    for block in range(4):
        # 14 steps per block: 7 for A_L/B_T, 7 for A_R/B_B
        for t in range(14):
            col = block * 14 + t
            
            if t < 7:
                # Use A_L (cols 0,1) and B_T (rows 0,1)
                u_vec = strassen_u[t]
                v_vec = strassen_v[t]
                w_vec = strassen_w[t]
                
                # A_L indices in U (size 8):
                # A is 2x4. A_L is cols 0,1.
                # Indices: 0,1 (row 0), 4,5 (row 1)
                U_mat[0][col] = u_vec[0]
                U_mat[1][col] = u_vec[1]
                U_mat[4][col] = u_vec[2]
                U_mat[5][col] = u_vec[3]
                
                # B_T indices in V (size 32):
                # B is 4x8. Block 'block' uses cols 2*block, 2*block+1.
                # B_T uses rows 0,1.
                # Indices: 0*8 + 2*block, 0*8 + 2*block+1, 1*8 + 2*block, 1*8 + 2*block+1
                base_col = 2 * block
                V_mat[0 * 8 + base_col + 0][col] = v_vec[0]
                V_mat[0 * 8 + base_col + 1][col] = v_vec[1]
                V_mat[1 * 8 + base_col + 0][col] = v_vec[2]
                V_mat[1 * 8 + base_col + 1][col] = v_vec[3]
                
                # C_block indices in W (size 16):
                # C is 2x8. Block 'block' uses cols 2*block, 2*block+1.
                # C is flattened k*n + i. k is col index 0..7. i is row index 0..1.
                # Indices: (2*block)*2 + 0, (2*block)*2 + 1, (2*block+1)*2 + 0, (2*block+1)*2 + 1
                W_mat[(2 * block) * 2 + 0][col] = w_vec[0]
                W_mat[(2 * block) * 2 + 1][col] = w_vec[1]
                W_mat[(2 * block + 1) * 2 + 0][col] = w_vec[2]
                W_mat[(2 * block + 1) * 2 + 1][col] = w_vec[3]
                
            else:
                # Use A_R (cols 2,3) and B_B (rows 2,3)
                t_idx = t - 7
                u_vec = strassen_u[t_idx]
                v_vec = strassen_v[t_idx]
                w_vec = strassen_w[t_idx]
                
                # A_R indices in U (size 8):
                # A_R is cols 2,3.
                # Indices: 2,3 (row 0), 6,7 (row 1)
                U_mat[2][col] = u_vec[0]
                U_mat[3][col] = u_vec[1]
                U_mat[6][col] = u_vec[2]
                U_mat[7][col] = u_vec[3]
                
                # B_B indices in V (size 32):
                # B_B uses rows 2,3.
                # Indices: 2*8 + 2*block, 2*8 + 2*block+1, 3*8 + 2*block, 3*8 + 2*block+1
                base_col = 2 * block
                V_mat[2 * 8 + base_col + 0][col] = v_vec[0]
                V_mat[2 * 8 + base_col + 1][col] = v_vec[1]
                V_mat[3 * 8 + base_col + 0][col] = v_vec[2]
                V_mat[3 * 8 + base_col + 1][col] = v_vec[3]
                
                # C_block indices in W (size 16):
                # Same C_block as before
                W_mat[(2 * block) * 2 + 0][col] = w_vec[0]
                W_mat[(2 * block) * 2 + 1][col] = w_vec[1]
                W_mat[(2 * block + 1) * 2 + 0][col] = w_vec[2]
                W_mat[(2 * block + 1) * 2 + 1][col] = w_vec[3]

    return {"U": U_mat, "V": V_mat, "W": W_mat}
