def construct():
    n, m, p = 4, 4, 6
    
    # Strassen 2x2x2
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 (indices k=0..3)
    # r = 7*H + L
    for H in range(7):
        for L in range(7):
            r = 7 * H + L
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        # Map to global indices
                        # i = 2*I + i0
                        # j = 2*J + j0
                        # k = 2*K + k0
                        # U row: i*m + j
                        # V row: j*p + k
                        # W row: k*n + i
                        # Factors:
                        # U_H[I*2+J, H] * U_L[i0*2+j0, L]
                        # V_H[J*2+K, H] * V_L[j0*2+k0, L]
                        # W_H[K*2+I, H] * W_L[k0*2+i0, L]
                        
                        u_h = S_U[I*2+J][H]
                        v_h = S_V[J*2+K][H]
                        w_h = S_W[K*2+I][H]
                        
                        for i0 in range(2):
                            for j0 in range(2):
                                for k0 in range(2):
                                    i = 2*I + i0
                                    j = 2*J + j0
                                    k = 2*K + k0
                                    
                                    u_l = S_U[i0*2+j0][L]
                                    v_l = S_V[j0*2+k0][L]
                                    w_l = S_W[k0*2+i0][L]
                                    
                                    U[i*m+j][r] = u_h * u_l
                                    V[j*p+k][r] = v_h * v_l
                                    W[k*n+i][r] = w_h * w_l
                                    
    # Part 2: 4x4x2 (indices k=4..5)
    # k = 4 + k0 where k0 in {0,1}
    # r = rank_444 + 7*(I*2+J) + L
    for I in range(2):
        for J in range(2):
            for L in range(7):
                r = rank_444 + 7*(I*2+J) + L
                for i0 in range(2):
                    for j0 in range(2):
                        for k0 in range(2):
                            i = 2*I + i0
                            j = 2*J + j0
                            k = 4 + k0
                            
                            u_l = S_U[i0*2+j0][L]
                            v_l = S_V[j0*2+k0][L]
                            w_l = S_W[k0*2+i0][L]
                            
                            U[i*m+j][r] = u_l
                            V[j*p+k][r] = v_l
                            W[k*n+i][r] = w_l
                            
    return {"U": U, "V": V, "W": W}
