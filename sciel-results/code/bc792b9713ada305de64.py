def get_strassen_decomp(n, p):
    # ... sub_decomps logic ...
    
    total_rank = sum(len(U[0]) for U, V, W in sub_decomps)
    
    U = [[0] * total_rank for _ in range(2 * n * 2 * n)]
    V = [[0] * total_rank for _ in range(2 * n * 2 * p)]
    W = [[0] * total_rank for _ in range(2 * p * 2 * n)]
    
    col = 0
    def add_sub(l, sign_u1, sign_u2, sign_v1, sign_v2, sign_w1, sign_w2, 
                u1_i, u1_j, u2_i, u2_j, 
                v1_j, v1_k, v2_j, v2_k, 
                w1_k, w1_i, w2_k, w2_i):
        nonlocal col
        U_sub, V_sub, W_sub = sub_decomps[l]
        rank_sub = len(U_sub[0])
        for r in range(rank_sub):
            # U
            for i in range(n):
                for j in range(n):
                    val = U_sub[i * n + j][r]
                    if sign_u1 != 0:
                        idx = (u1_i + i) * (2 * n) + (u1_j + j)
                        U[idx][col] += sign_u1 * val
                    if sign_u2 != 0:
                        idx = (u2_i + i) * (2 * n) + (u2_j + j)
                        U[idx][col] += sign_u2 * val
                        
            # V
            for j in range(n):
                for k in range(p):
                    val = V_sub[j * p + k][r]
                    if sign_v1 != 0:
                        idx = (v1_j + j) * (2 * p) + (v1_k + k)
                        V[idx][col] += sign_v1 * val
                    if sign_v2 != 0:
                        idx = (v2_j + j) * (2 * p) + (v2_k + k)
                        V[idx][col] += sign_v2 * val
                        
            # W
            for k in range(p):
                for i in range(n):
                    val = W_sub[k * n + i][r]
                    if sign_w1 != 0:
                        idx = (w1_k + k) * (2 * n) + (w1_i + i)
                        W[idx][col] += sign_w1 * val
                    if sign_w2 != 0:
                        idx = (w2_k + k) * (2 * n) + (w2_i + i)
                        W[idx][col] += sign_w2 * val
                        
            col += 1

    # ... calls to add_sub ...
    return U, V, W
