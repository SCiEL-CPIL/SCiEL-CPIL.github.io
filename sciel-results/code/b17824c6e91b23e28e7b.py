import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    rank = 49 + 4 + 4 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def get_strassen_2x2():
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        u[0][0]=1; u[3][0]=1; v[0][0]=1; v[3][0]=1; w[0][0]=1; w[3][0]=1
        u[2][1]=1; u[3][1]=1; v[0][1]=1; w[2][1]=1; w[3][1]=-1
        u[0][2]=1; v[1][2]=1; v[3][2]=-1; w[1][2]=1; w[3][2]=1
        u[3][3]=1; v[2][3]=1; v[0][3]=-1; w[0][3]=1; w[2][3]=1
        u[0][4]=1; u[1][4]=1; v[3][4]=1; w[0][4]=-1; w[1][4]=1
        u[0][5]=-1; u[2][5]=1; v[0][5]=1; v[1][5]=1; w[3][5]=1
        u[1][6]=1; u[3][6]=-1; v[2][6]=1; v[3][6]=1; w[0][6]=1
        return u, v, w
    
    s_u, s_v, s_w = get_strassen_2x2()
    
    def add_4x4_term(a_coeffs, b_coeffs, c_coeffs, start_col):
        for inner_k in range(7):
            col_idx = start_col + inner_k
            u_vec = [s_u[r][inner_k] for r in range(4)]
            v_vec = [s_v[r][inner_k] for r in range(4)]
            w_vec = [s_w[r][inner_k] for r in range(4)]
            
            for (a_br, a_bc), a_c in a_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        if u_vec[li] != 0:
                            gi = 2 * a_br + lr
                            gj = 2 * a_bc + lc
                            U[gi * 5 + gj][col_idx] += a_c * u_vec[li]
            
            for (b_br, b_bc), b_c in b_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        if v_vec[li] != 0:
                            gj = 2 * b_br + lr
                            gk = 2 * b_bc + lc
                            V[gj * 5 + gk][col_idx] += b_c * v_vec[li]
            
            for (c_br, c_bc), c_c in c_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        if w_vec[li] != 0:
                            gk = 2 * c_bc + lc
                            gi = 2 * c_br + lr
                            W[gk * 4 + gi][col_idx] += c_c * w_vec[li]
    
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for t_idx, (a_c, b_c, c_c) in enumerate(terms):
        add_4x4_term(a_c, b_c, c_c, t_idx * 7)
    
    col = 49
    
    # 4x4x1: rank 4
    for r in range(4):
        for i in range(4):
            for k in range(4):
                if i == r:
                    U[i*5 + k][col] = 1
                    V[k*5 + 4][col] = 1
                    W[16 + i][col] = 1
        col += 1
    
    # 4x1x4: rank 4
    for r in range(4):
        for i in range(4):
            for j in range(4):
                if i == r:
                    U[i*5 + 4][col] = 1
                    V[20 + j][col] = 1
                    W[j*4 + i][col] = 1
        col += 1
    
    # 4x1x1: rank 4
    for r in range(4):
        for i in range(4):
            if i == r:
                U[i*5 + 4][col] = 1
                V[24][col] = 1
                W[16 + i][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
