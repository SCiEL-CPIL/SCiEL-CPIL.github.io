def construct():
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(rank_offset, j_offset, k_offset):
        a_idx = 0 * m + j_offset
        b_idx = 0 * m + (j_offset + 1)
        c_idx = 1 * m + j_offset
        d_idx = 1 * m + (j_offset + 1)
        
        e_idx = j_offset * p + k_offset
        f_idx = j_offset * p + (k_offset + 1)
        g_idx = (j_offset + 1) * p + k_offset
        h_idx = (j_offset + 1) * p + (k_offset + 1)
        
        r0 = rank_offset
        r1 = rank_offset + 1
        r2 = rank_offset + 2
        r3 = rank_offset + 3
        r4 = rank_offset + 4
        r5 = rank_offset + 5
        r6 = rank_offset + 6
        
        U[a_idx][r0] += 1; U[d_idx][r0] += 1
        U[c_idx][r1] += 1; U[d_idx][r1] += 1
        U[a_idx][r2] += 1
        U[d_idx][r3] += 1
        U[a_idx][r4] += 1; U[b_idx][r4] += 1
        U[c_idx][r5] += 1; U[a_idx][r5] -= 1
        U[b_idx][r6] += 1; U[d_idx][r6] -= 1
        
        V[e_idx][r0] += 1; V[h_idx][r0] += 1
        V[e_idx][r1] += 1
        V[f_idx][r2] += 1; V[h_idx][r2] -= 1
        V[g_idx][r3] += 1; V[e_idx][r3] -= 1
        V[h_idx][r4] += 1
        V[e_idx][r5] += 1; V[f_idx][r5] += 1
        V[g_idx][r6] += 1; V[h_idx][r6] += 1
        
        W[k_offset * n + 0][r0] += 1
        W[k_offset * n + 0][r3] += 1
        W[k_offset * n + 0][r4] -= 1
        W[k_offset * n + 0][r6] += 1
        
        W[(k_offset + 1) * n + 0][r2] += 1
        W[(k_offset + 1) * n + 0][r4] += 1
        
        W[k_offset * n + 1][r1] += 1
        W[k_offset * n + 1][r3] += 1
        
        W[(k_offset + 1) * n + 1][r0] += 1
        W[(k_offset + 1) * n + 1][r1] -= 1
        W[(k_offset + 1) * n + 1][r2] += 1
        W[(k_offset + 1) * n + 1][r5] += 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(7, 0, 2)
    
    r0, r1, r2, r3 = 14, 15, 16, 17
    U[0][r0] += 1
    U[1][r1] += 1
    U[3][r2] += 1
    U[4][r3] += 1
    
    V[4][r0] += 1
    V[9][r1] += 1
    V[4][r2] += 1
    V[9][r3] += 1
    
    W[8][r0] += 1
    W[8][r1] += 1
    W[9][r2] += 1
    W[9][r3] += 1
    
    rank_offset = 18
    for k in range(5):
        r_u0 = rank_offset
        r_u1 = rank_offset + 1
        rank_offset += 2
        
        U[2][r_u0] += 1
        U[5][r_u1] += 1
        
        V[10 + k][r_u0] += 1
        V[10 + k][r_u1] += 1
        
        W[k * n + 0][r_u0] += 1
        W[k * n + 1][r_u1] += 1
        
    return {"U": U, "V": V, "W": W}

BET: {mechanism: "The previous response failed validation due to non-code content in the output. This candidate provides a clean, executable implementation of the same structural approach (Strassen-based 28-rank decomposition) without extraneous text, ensuring it is admitted for evaluation.", q: 0.9, delta: [0.0, 0.0]}
