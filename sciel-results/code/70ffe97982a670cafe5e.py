import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen's 2x2x2 decomposition vectors
    # u, v are length 4 (flattened 2x2 matrices)
    # w is length 4 (flattened 2x2 output coefficients)
    
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]
    
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, 0]
    
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, -1]
    
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [-1, 1, 0, 0]
    
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [1, 0, 0, 0]
    
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]
    
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 1, 0]
    
    us = [u1, u2, u3, u4, u5, u6, u7]
    vs = [v1, v2, v3, v4, v5, v6, v7]
    ws = [w1, w2, w3, w4, w5, w6, w7]
    
    # Generate 4x4x4 decomposition (rank 49)
    # This corresponds to M_4,4,4
    # u_st = u_s (x) u_t (length 16)
    # v_st = v_s (x) v_t (length 16)
    # w_st = w_s (x) w_t (length 16)
    
    U_444 = []
    V_444 = []
    W_444 = []
    
    for s in range(7):
        for t in range(7):
            u_st = np.kron(us[s], us[t])
            v_st = np.kron(vs[s], vs[t])
            w_st = np.kron(ws[s], ws[t])
            U_444.append(u_st)
            V_444.append(v_st)
            W_444.append(w_st)
            
    # Generate 4x4x1 decomposition (rank 16)
    # Schoolbook for M_4,4,1
    U_441 = []
    V_441 = []
    W_441 = []
    
    for i in range(4):
        for j in range(4):
            u = [0] * 16
            u[i * 4 + j] = 1
            U_441.append(u)
            
            v = [0] * 4 # m*p = 4*1 = 4
            v[j] = 1
            V_441.append(v)
            
            w = [0] * 4 # p*n = 1*4 = 4
            w[i] = 1
            W_441.append(w)
            
    # Total rank
    rank = len(U_444) * 2 + len(U_441)
    
    U = [0] * (n * m) # 16
    V = [0] * (m * p) # 36
    W = [0] * (p * n) # 36
    
    # Initialize matrices as lists of lists for the return value
    # U: 16 x rank
    # V: 36 x rank
    # W: 36 x rank
    
    U_mat = [[0] * rank for _ in range(16)]
    V_mat = [[0] * rank for _ in range(36)]
    W_mat = [[0] * rank for _ in range(36)]
    
    col = 0
    
    # Block 1: B cols 0-3
    for i in range(len(U_444)):
        u_vec = U_444[i]
        v_vec = V_444[i]
        w_vec = W_444[i]
        
        for row in range(16):
            U_mat[row][col] = u_vec[row]
            
        for row in range(16):
            # v_vec index: j*4 + k. Here k is 0..3
            V_mat[row][col] = v_vec[row]
            
        for row in range(16):
            # w_vec index: k*4 + i. Here k is 0..3
            W_mat[row][col] = w_vec[row]
            
        col += 1
        
    # Block 2: B cols 4-7
    for i in range(len(U_444)):
        u_vec = U_444[i]
        v_vec = V_444[i]
        w_vec = W_444[i]
        
        for row in range(16):
            U_mat[row][col] = u_vec[row]
            
        for row in range(16):
            # v_vec index j*4 + k. We want k=4..7.
            # So we map v_vec[j*4 + k'] to V_mat[j*4 + (k'+4)]
            # v_vec is length 16, indices 0..15.
            # row in 0..15.
            # j = row // 4, k' = row % 4.
            # target index = j*4 + (k'+4) = row + 16.
            V_mat[row + 16][col] = v_vec[row]
            
        for row in range(16):
            # w_vec index k*4 + i. We want k=4..7.
            # w_vec is length 16.
            # row in 0..15.
            # k = row // 4, i = row % 4.
            # target index = (k+4)*4 + i = row + 16.
            W_mat[row + 16][col] = w_vec[row]
            
        col += 1
        
    # Block 3: B col 8
    for i in range(len(U_441)):
        u_vec = U_441[i]
        v_vec = V_441[i]
        w_vec = W_441[i]
        
        for row in range(16):
            U_mat[row][col] = u_vec[row]
            
        for row in range(4):
            # v_vec index j. We want k=8.
            # target index = j*4 + 8 = row + 32.
            V_mat[row + 32][col] = v_vec[row]
            
        for row in range(4):
            # w_vec index i. We want k=8.
            # target index = 8*4 + i = 32 + row.
            W_mat[row + 32][col] = w_vec[row]
            
        col += 1
        
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
