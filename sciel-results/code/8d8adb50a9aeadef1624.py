import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Standard Strassen coefficients for 2x2
    # c[s][i][j] is coeff of A[i][j] in A_s
    # d[s][j][k] is coeff of B[j][k] in B_s
    # e[s][i][k] is coeff of M_s in C[i][k]
    
    c = [[0]*2 for _ in range(2)]
    d = [[0]*2 for _ in range(2)]
    e = [[0]*2 for _ in range(2)]
    
    # We need 7 sets of coefficients
    # Let's store them in lists of dicts or 3D lists
    # c_coeffs[s][i][j]
    c_coeffs = [[[[0]*2 for _ in range(2)] for _ in range(7)] for _ in range(7)] 
    # This is getting messy. Let's just define the 7 cases explicitly.
    
    # Case 1: (A00+A11)(B00+B11)
    # Case 2: (A10+A11)B00
    # Case 3: A00(B01-B11)
    # Case 4: A11(B10-B00)
    # Case 5: (A00+A01)B11
    # Case 6: (A10-A00)(B00+B01)
    # Case 7: (A01-A11)(B10+B11)
    
    def get_strassen_coeffs(s):
        # Returns c, d, e matrices for the s-th Strassen term (0-indexed)
        c_mat = [[0]*2 for _ in range(2)]
        d_mat = [[0]*2 for _ in range(2)]
        e_mat = [[0]*2 for _ in range(2)]
        
        if s == 0:
            # M1
            c_mat[0][0] = 1; c_mat[1][1] = 1
            d_mat[0][0] = 1; d_mat[1][1] = 1
            e_mat[0][0] = 1; e_mat[1][1] = 1
            e_mat[0][1] = 0; e_mat[1][0] = 0
            # C00 += M1, C11 += M1
        elif s == 1:
            # M2
            c_mat[1][0] = 1; c_mat[1][1] = 1
            d_mat[0][0] = 1
            e_mat[1][0] = 1; e_mat[1][1] = -1
            # C10 += M2, C11 -= M2
        elif s == 2:
            # M3
            c_mat[0][0] = 1
            d_mat[0][1] = 1; d_mat[1][1] = -1
            e_mat[0][1] = 1; e_mat[1][1] = 1
            # C01 += M3, C11 += M3
        elif s == 3:
            # M4
            c_mat[1][1] = 1
            d_mat[1][0] = 1; d_mat[0][0] = -1
            e_mat[0][0] = 1; e_mat[1][0] = 1
            # C00 += M4, C10 += M4
        elif s == 4:
            # M5
            c_mat[0][0] = 1; c_mat[0][1] = 1
            d_mat[1][1] = 1
            e_mat[0][0] = -1; e_mat[0][1] = 1
            # C00 -= M5, C01 += M5
        elif s == 5:
            # M6
            c_mat[1][0] = 1; c_mat[0][0] = -1
            d_mat[0][0] = 1; d_mat[0][1] = 1
            e_mat[1][1] = 1
            # C11 += M6
        elif s == 6:
            # M7
            c_mat[0][1] = 1; c_mat[1][1] = -1
            d_mat[1][0] = 1; d_mat[1][1] = 1
            e_mat[0][0] = 1
            # C00 += M7
            
        return c_mat, d_mat, e_mat

    # Precompute coefficients for all 7 top-level terms
    top_c = []
    top_d = []
    top_e = []
    for s in range(7):
        c_mat, d_mat, e_mat = get_strassen_coeffs(s)
        top_c.append(c_mat)
        top_d.append(d_mat)
        top_e.append(e_mat)

    # Precompute coefficients for all 7 bottom-level terms (same as top)
    bot_c = top_c
    bot_d = top_d
    bot_e = top_e

    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]

    for s in range(7):
        for k in range(7):
            r = s * 7 + k
            
            # U coefficients
            # U[i*4+j, r] = sum_{p,q} bot_c[k][p][q] * top_c[s][I][J]
            # where i = 2I+p, j = 2J+q
            for I in range(2):
                for J in range(2):
                    top_coeff = top_c[s][I][J]
                    if top_coeff == 0:
                        continue
                    for p in range(2):
                        for q in range(2):
                            bot_coeff = bot_c[k][p][q]
                            if bot_coeff == 0:
                                continue
                            i = 2 * I + p
                            j = 2 * J + q
                            u_idx = i * 4 + j
                            U[u_idx][r] += top_coeff * bot_coeff

            # V coefficients
            # V[j*4+k, r] = sum_{r_loc, t_loc} bot_d[k][r_loc][t_loc] * top_d[s][J_prime][K]
            # where j = 2J_prime + r_loc, k = 2K + t_loc
            for J_prime in range(2):
                for K in range(2):
                    top_coeff = top_d[s][J_prime][K]
                    if top_coeff == 0:
                        continue
                    for r_loc in range(2):
                        for t_loc in range(2):
                            bot_coeff = bot_d[k][r_loc][t_loc]
                            if bot_coeff == 0:
                                continue
                            j = 2 * J_prime + r_loc
                            k = 2 * K + t_loc
                            v_idx = j * 4 + k
                            V[v_idx][r] += top_coeff * bot_coeff

            # W coefficients
            # W[k*4+i, r] = top_e[s][P][Q] * bot_e[k][p_loc][q_loc]
            # where i = 2P + p_loc, k = 2Q + q_loc
            for P in range(2):
                for Q in range(2):
                    top_coeff = top_e[s][P][Q]
                    if top_coeff == 0:
                        continue
                    for p_loc in range(2):
                        for q_loc in range(2):
                            bot_coeff = bot_e[k][p_loc][q_loc]
                            if bot_coeff == 0:
                                continue
                            i = 2 * P + p_loc
                            k = 2 * Q + q_loc
                            w_idx = k * 4 + i
                            W[w_idx][r] += top_coeff * bot_coeff

    return {"U": U, "V": V, "W": W}

