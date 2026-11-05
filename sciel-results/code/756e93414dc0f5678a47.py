import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_rank = 117
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Partition definitions
    # Indices 0,1 -> block 0
    # Indices 2,3 -> block 1
    # Indices 4 -> block 2
    def get_block(idx):
        if idx < 2:
            return 0, idx
        elif idx < 4:
            return 1, idx - 2
        else:
            return 2, idx - 4

    def get_global(block, loc):
        if block == 0:
            return loc
        elif block == 1:
            return loc + 2
        else:
            return loc + 4

    block_sizes = [2, 2, 1]
    
    current_rank = 0
    
    # Precompute Strassen coefficients for 2x2x2
    # U, V, W are 4x7, 4x7, 4x7 (flattened 2x2)
    # Indices: 00, 01, 10, 11
    strassen_U = [
        [1, 0, 0, 0, 0, 0, 0], # m1: a
        [0, 0, 1, 0, 0, -1, 0], # m1: b? No, m1 is a+d.
        # Let's re-verify Strassen U, V, W
        # m1 = (a+d)(e+h)
        # U1: a=1, d=1 -> [1, 0, 0, 1]
        # V1: e=1, h=1 -> [1, 0, 0, 1]
        # W1: C00=1, C11=1 -> [1, 0, 0, 1] (W is ki, so 00, 01, 10, 11)
        
        # Let's build them properly
    ]
    
    def get_strassen():
        # U[i,j,r], V[j,k,r], W[k,i,r]
        # i,j,k in {0,1}
        # r in {0..6}
        U_loc = [[0]*4 for _ in range(4)] # flattened 2x2
        V_loc = [[0]*4 for _ in range(4)]
        W_loc = [[0]*4 for _ in range(4)]
        
        # m1 = (a+d)(e+h)
        U_loc[0][0] = 1; U_loc[3][0] = 1 # a, d
        V_loc[0][0] = 1; V_loc[3][0] = 1 # e, h
        W_loc[0][0] = 1; W_loc[3][0] = 1 # C00, C11
        
        # m2 = (c+d)e
        U_loc[2][1] = 1; U_loc[3][1] = 1 # c, d
        V_loc[0][1] = 1 # e
        W_loc[2][1] = 1; W_loc[3][1] = -1 # C10, C11
        
        # m3 = a(f-h)
        U_loc[0][2] = 1 # a
        V_loc[1][2] = 1; V_loc[3][2] = -1 # f, h
        W_loc[1][2] = 1; W_loc[3][2] = 1 # C01, C11
        
        # m4 = d(g-f)
        U_loc[3][3] = 1 # d
        V_loc[2][3] = 1; V_loc[1][3] = -1 # g, f
        W_loc[0][3] = 1; W_loc[2][3] = 1 # C00, C10
        
        # m5 = (a+b)h
        U_loc[0][4] = 1; U_loc[1][4] = 1 # a, b
        V_loc[3][4] = 1 # h
        W_loc[0][4] = -1; W_loc[1][4] = 1 # C00, C01
        
        # m6 = (c-a)(e+f)
        U_loc[2][5] = 1; U_loc[0][5] = -1 # c, a
        V_loc[0][5] = 1; V_loc[1][5] = 1 # e, f
        W_loc[3][5] = 1 # C11
        
        # m7 = (b-d)(g+h)
        U_loc[1][6] = 1; U_loc[3][6] = -1 # b, d
        V_loc[2][6] = 1; V_loc[3][6] = 1 # g, h
        W_loc[0][6] = 1 # C00
        
        return U_loc, V_loc, W_loc

    strassen_U, strassen_V, strassen_W = get_strassen()
    
    for alpha in range(3):
        for beta in range(3):
            for gamma in range(3):
                n_a = block_sizes[alpha]
                m_b = block_sizes[beta]
                p_g = block_sizes[gamma]
                
                if n_a == 2 and m_b == 2 and p_g == 2:
                    rank_loc = 7
                else:
                    rank_loc = n_a * m_b * p_g
                
                for r_loc in range(rank_loc):
                    r_glob = current_rank + r_loc
                    
                    if n_a == 2 and m_b == 2 and p_g == 2:
                        # Strassen
                        for i_loc in range(2):
                            for j_loc in range(2):
                                idx_u = i_loc * 2 + j_loc
                                val = strassen_U[idx_u][r_loc]
                                if val != 0:
                                    i_glob = get_global(alpha, i_loc)
                                    j_glob = get_global(beta, j_loc)
                                    U[i_glob * m + j_glob][r_glob] += val
                                    
                        for j_loc in range(2):
                            for k_loc in range(2):
                                idx_v = j_loc * 2 + k_loc
                                val = strassen_V[idx_v][r_loc]
                                if val != 0:
                                    j_glob = get_global(beta, j_loc)
                                    k_glob = get_global(gamma, k_loc)
                                    V[j_glob * p + k_glob][r_glob] += val
                                    
                        for k_loc in range(2):
                            for i_loc in range(2):
                                idx_w = k_loc * 2 + i_loc
                                val = strassen_W[idx_w][r_loc]
                                if val != 0:
                                    k_glob = get_global(gamma, k_loc)
                                    i_glob = get_global(alpha, i_loc)
                                    W[k_glob * n + i_glob][r_glob] += val
                    else:
                        # Schoolbook
                        # Rank is n_a * m_b * p_g
                        # Each rank component corresponds to a single (i_loc, j_loc, k_loc)
                        # We need to map r_loc to (i_loc, j_loc, k_loc)
                        # Let's iterate i_loc, j_loc, k_loc
                        # r_loc = i_loc * m_b * p_g + j_loc * p_g + k_loc
                        i_loc = r_loc // (m_b * p_g)
                        rem = r_loc % (m_b * p_g)
                        j_loc = rem // p_g
                        k_loc = rem % p_g
                        
                        i_glob = get_global(alpha, i_loc)
                        j_glob = get_global(beta, j_loc)
                        k_glob = get_global(gamma, k_loc)
                        
                        U[i_glob * m + j_glob][r_glob] = 1
                        V[j_glob * p + k_glob][r_glob] = 1
                        W[k_glob * n + i_glob][r_glob] = 1
                        
                current_rank += rank_loc
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
