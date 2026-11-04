import numpy as np

def construct():
    # Dimensions
    n, m, p = 4, 4, 6
    
    # Sub-problem 1: 2x2x2 (Strassen, Rank 7)
    # U_A: 4x7, V_A: 4x7, W_A: 4x7
    # Indices: U: i*2+j, V: j*2+k, W: k*2+i
    
    U_A = [
        [1, 0, 1, 0, 1, 0, 0], # a
        [0, 0, 0, 0, 1, 0, 1], # b
        [0, 1, 0, 0, 0, 1, 0], # c
        [1, 1, 0, 1, 0, -1, -1] # d
    ]
    
    V_A = [
        [1, 1, 0, 0, 0, 1, 0], # x
        [0, 0, 1, -1, 0, 1, 0], # y
        [0, 0, 0, 1, 0, 0, 1], # z
        [1, 0, 1, 0, 1, 0, 1]  # w
    ]
    
    W_A = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, -1, 1, -1, -1], # C01
        [0, 1, 0, 1, 0, 1, -1], # C10
        [1, -1, 1, 0, 0, 0, 1]  # C11
    ]
    
    # Sub-problem 2: 2x2x3 (Schoolbook, Rank 12)
    # U_B: 4x12, V_B: 6x12, W_B: 6x12
    # Indices: U_B: i*2+j (4 rows), V_B: j*3+k (6 rows), W_B: k*2+i (6 rows)
    
    U_B = [[0]*12 for _ in range(4)]
    V_B = [[0]*12 for _ in range(6)]
    W_B = [[0]*12 for _ in range(6)]
    
    # a=0, b=1, c=2, d=3
    # x=0, y=1, z=2, u=3, v=4, w=5
    # C00=0, C01=1, C02=2, C10=3, C11=4, C12=5
    
    terms = [
        (0, 0, 0), # a*x -> C00
        (0, 1, 1), # a*y -> C01
        (0, 2, 2), # a*z -> C02
        (1, 3, 0), # b*u -> C00
        (1, 4, 1), # b*v -> C01
        (1, 5, 2), # b*w -> C02
        (2, 0, 3), # c*x -> C10
        (2, 1, 4), # c*y -> C11
        (2, 2, 5), # c*z -> C12
        (3, 3, 3), # d*u -> C10
        (3, 4, 4), # d*v -> C11
        (3, 5, 5)  # d*w -> C12
    ]
    
    for idx, (u_idx, v_idx, w_idx) in enumerate(terms):
        U_B[u_idx][idx] = 1
        V_B[v_idx][idx] = 1
        W_B[w_idx][idx] = 1
        
    rank_A = 7
    rank_B = 12
    total_rank = rank_A * rank_B
    
    U = [[0]*total_rank for _ in range(n*m)] # 16 x 84
    V = [[0]*total_rank for _ in range(m*p)] # 24 x 84
    W = [[0]*total_rank for _ in range(p*n)] # 24 x 84
    
    for r_A in range(rank_A):
        for r_B in range(rank_B):
            col = r_A * rank_B + r_B
            
            # Fill U
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            row_U = (i1*2 + i2)*4 + (j1*2 + j2)
                            u_A_val = U_A[i1*2 + j1][r_A]
                            u_B_val = U_B[i2*2 + j2][r_B]
                            U[row_U][col] = u_A_val * u_B_val
                            
            # Fill V
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            row_V = (j1*2 + j2)*6 + (k1*2 + k2)
                            v_A_val = V_A[j1*2 + k1][r_A]
                            v_B_val = V_B[j2*3 + k2][r_B]
                            V[row_V][col] = v_A_val * v_B_val
                            
            # Fill W
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            row_W = (k1*2 + k2)*4 + (i1*2 + i2)
                            w_A_val = W_A[k1*2 + i1][r_A]
                            w_B_val = W_B[k2*2 + i2][r_B]
                            W[row_W][col] = w_A_val * w_B_val
                            
    return {"U": U, "V": V, "W": W}
