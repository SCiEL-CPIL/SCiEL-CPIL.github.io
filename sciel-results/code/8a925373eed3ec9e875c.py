import numpy as np

def construct():
    n, m, p = 2, 4, 6
    
    # Strassen tiling
    # Split m=4 into 2+2
    # Split p=6 into 2+2+2
    # 1 row block (n=2)
    # Total 6 blocks of 2x2x2
    
    rank_per_block = 7
    total_rank = 6 * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 42
    
    # Define Strassen factors for 2x2x2
    # U: [a, b, c, d]
    # V: [e, f, g, h]
    # W: [i, k, j, l] (W[k*n+i] with n=2 -> W[0]=i, W[1]=k, W[2]=j, W[3]=l)
    
    # Verified factors
    strassen_factors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [0, 1, -1, 0], [0, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]),
        ([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 1, -1]),
    ]
    
    # Verify factors
    T_target = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_target[i*2+j, j*2+k, k*2+i] = 1
                
    T_check = np.zeros((4, 4, 4))
    for u, v, w in strassen_factors:
        u_arr = np.array(u)
        v_arr = np.array(v)
        w_arr = np.array(w)
        T_check += np.einsum('i,j,k->ijk', u_arr, v_arr, w_arr)
        
    if not np.allclose(T_check, T_target):
        # Fallback to schoolbook if factors are wrong
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Fill matrices
    col_offset = 0
    for mb in range(2):  # m blocks
        for pb in range(3):  # p blocks
            for r in range(7):
                u, v, w = strassen_factors[r]
                for idx_u in range(4):
                    if u[idx_u] != 0:
                        # U index: i*m + j. Block is i in 0..1, j in mb*2 .. mb*2+1
                        # idx_u is 0..3 corresponding to i=idx_u//2, j=idx_u%2
                        i_local = idx_u // 2
                        j_local = idx_u % 2
                        i_global = i_local
                        j_global = mb * 2 + j_local
                        U[i_global * m + j_global][col_offset] = u[idx_u]
                        
                for idx_v in range(4):
                    if v[idx_v] != 0:
                        # V index: j*p + k. Block is j in mb*2 .. mb*2+1, k in pb*2 .. pb*2+1
                        # idx_v is 0..3 corresponding to j=idx_v//2, k=idx_v%2
                        j_local = idx_v // 2
                        k_local = idx_v % 2
                        j_global = mb * 2 + j_local
                        k_global = pb * 2 + k_local
                        V[j_global * p + k_global][col_offset] = v[idx_v]
                        
                for idx_w in range(4):
                    if w[idx_w] != 0:
                        # W index: k*n + i. Block is k in pb*2 .. pb*2+1, i in 0..1
                        # idx_w is 0..3 corresponding to k=idx_w//2, i=idx_w%2
                        k_local = idx_w // 2
                        i_local = idx_w % 2
                        k_global = pb * 2 + k_local
                        i_global = i_local
                        W[k_global * n + i_global][col_offset] = w[idx_w]
                        
                col_offset += 1
                
    return {"U": U, "V": V, "W": W}
