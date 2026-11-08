import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get 2x2x2 Strassen factors
    def get_strassen_2x2_factors():
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        u[0][0]=1; u[3][0]=1
        v[0][0]=1; v[3][0]=1
        w[0][0]=1; w[3][0]=1
        
        u[2][1]=1; u[3][1]=1
        v[0][1]=1
        w[2][1]=1; w[3][1]=-1
        
        u[0][2]=1
        v[1][2]=1; v[3][2]=-1
        w[1][2]=1; w[3][2]=1
        
        u[3][3]=1
        v[2][3]=1; v[0][3]=-1
        w[0][3]=1; w[2][3]=1
        
        u[0][4]=1; u[1][4]=1
        v[3][4]=1
        w[0][4]=-1; w[1][4]=1
        
        u[0][5]=-1; u[2][5]=1
        v[0][5]=1; v[1][5]=1
        w[3][5]=1
        
        u[1][6]=1; u[3][6]=-1
        v[2][6]=1; v[3][6]=1
        w[0][6]=1
        
        return u, v, w

    s_u, s_v, s_w = get_strassen_2x2_factors()
    
    def add_4x4_term(a_block_coeffs, b_block_coeffs, c_block_coeffs, start_col):
        for inner_k in range(7):
            u_vec = [s_u[r][inner_k] for r in range(4)]
            v_vec = [s_v[r][inner_k] for r in range(4)]
            w_vec = [s_w[r][inner_k] for r in range(4)]
            
            col_idx = start_col + inner_k
            
            for (a_br, a_bc), a_coeff in a_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            global_i = 2 * a_br + lr
                            global_j = 2 * a_bc + lc
                            global_u_row = global_i * 5 + global_j
                            U[global_u_row][col_idx] += a_coeff * u_val
            
            for (b_br, b_bc), b_coeff in b_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            global_j = 2 * b_br + lr
                            global_k = 2 * b_bc + lc
                            global_v_row = global_j * 5 + global_k
                            V[global_v_row][col_idx] += b_coeff * v_val
            
            for (c_br, c_bc), c_coeff in c_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            global_k = 2 * c_bc + lc
                            global_i = 2 * c_br + lr
                            global_w_row = global_k * 4 + global_i
                            W[global_w_row][col_idx] += c_coeff * w_val

    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        start_col = t_idx * 7
        add_4x4_term(a_coeff, b_coeff, c_coeff, start_col)
        
    col = 49
    
    # 1. A_41 B_14 (4x1 * 1x4 -> 4x4) Rank 4
    # U[i, 4] * V[4, j] -> W[j, i] ? No.
    # C[i, j] += A[i, 4] B[4, j]
    # U[i*5+4, col] = 1
    # V[4*5+j, col] = 1
    # W[j*4+i, col] = 1
    # This is rank 16 if done naively.
    # We can do rank 4:
    # Sum_r U[i, 4, r] V[4, j, r] W[j, i, r] = A[i, 4] B[4, j]
    # Let U[i, 4, r] = 1 if r==i else 0?
    # Let V[4, j, r] = 1 if r==j else 0?
    # Then sum is 1 if i==j else 0. No.
    # We need A[i,4] B[4,j].
    # Set U[i, 4, r] = A[i,4] for all r? No.
    # Set U[i, 4, r] = A[i,4], V[4, j, r] = B[4,j], W[j, i, r] = 1/4.
    # Sum is 4 * A[i,4] B[4,j] * 1/4 = A[i,4] B[4,j].
    # This uses 4 columns.
    for r in range(4):
        for i in range(4):
            U[i*5 + 4][col] = 1
        for j in range(4):
            V[4*5 + j][col] = 1
        for j in range(4):
            for i in range(4):
                W[j*4 + i][col] = 0.25
        col += 1

    # 2. A_44 B_41 (4x4 * 4x1 -> 4x1) Rank 4
    # C[i, 4] += sum_k A[i, k] B[k, 4]
    # This is a vector-matrix product.
    # Can be done with rank 4?
    # C[i, 4] = sum_k A[i, k] B[k, 4]
    # Decomposition: sum_r U[i, k, r] V[k, 4, r] W[4, i, r]
    # This is more complex.
    # Actually, A_44 B_41 is 4x4 * 4x1.
    # Rank of 4x4x1 tensor is 4.
    # U[i, k, r], V[k, 4, r], W[4, i, r].
    # We can set W[4, i, r] = 1 if r==i else 0.
    # Then sum_r U[i, k, r] V[k, 4, r] = A[i, k] B[k, 4].
    # This requires U[i, k, r] = A[i, k] delta_{ir}? No.
    # If W[4, i, r] = delta_{ir}, then sum is U[i, k, i] V[k, 4, i].
    # We want sum_k A[i, k] B[k, 4].
    # So set U[i, k, i] = A[i, k], V[k, 4, i] = B[k, 4].
    # This works.
    for r in range(4):
        for i in range(4):
            if i == r:
                for k in range(4):
                    U[i*5 + k][col] = 1
                    V[k*5 + 4][col] = 1
                    W[4*4 + i][col] = 1
        col += 1

    # 3. A_41 B_11 (4x1 * 1x1 -> 4x1) Rank 1
    # C[i, 4] += A[i, 4] B[4, 4]
    U[0*5 + 4][col] = 1 # Placeholder, will scale
    # Actually:
    for i in range(4):
        U[i*5 + 4][col] = 1
    V[4*5 + 4][col] = 1
    for i in range(4):
        W[4*4 + i][col] = 1
    col += 1

    # 4. A_14 B_44 (1x4 * 4x4 -> 1x4) Rank 4
    # C[4, j] += sum_k A[4, k] B[k, j]
    # W[j, 4, r] ? No, W is p x n = 5 x 4.
    # W index is k*n + i. Here k=4, i=j.
    # W[4*4 + j].
    for r in range(4):
        for k in range(4):
            if k == r:
                U[4*5 + k][col] = 1
                for j in range(4):
                    V[k*5 + j][col] = 1
                    W[4*4 + j][col] = 1
        col += 1

    # 5. A_14 B_41 (1x4 * 4x1 -> 1x1) Rank 1
    # C[4, 4] += sum_k A[4, k] B[k, 4]
    for k in range(4):
        U[4*5 + k][col] = 1
        V[k*5 + 4][col] = 1
    W[4*4 + 4][col] = 1
    col += 1

    # 6. A_11 B_14 (1x1 * 1x4 -> 1x4) Rank 1
    # C[4, j] += A[4, 4] B[4, j]
    U[4*5 + 4][col] = 1
    for j in range(4):
        V[4*5 + j][col] = 1
        W[4*4 + j][col] = 1
    col += 1

    # 7. A_11 B_11 (1x1 * 1x1 -> 1x1) Rank 1
    # C[4, 4] += A[4, 4] B[4, 4]
    U[4*5 + 4][col] = 1
    V[4*5 + 4][col] = 1
    W[4*4 + 4][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}
