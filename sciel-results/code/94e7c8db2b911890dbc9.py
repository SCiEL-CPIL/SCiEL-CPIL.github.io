    def add_strassen(j_offset, k_offset):
        nonlocal col
        
        u_idx_a = 0*m + j_offset
        u_idx_b = 0*m + j_offset + 1
        u_idx_c = 1*m + j_offset
        u_idx_d = 1*m + j_offset + 1
        
        v_idx_e = j_offset*p + k_offset
        v_idx_f = j_offset*p + k_offset + 1
        v_idx_g = (j_offset+1)*p + k_offset
        v_idx_h = (j_offset+1)*p + k_offset + 1
        
        w_idx_c11 = k_offset*n + 0
        w_idx_c12 = (k_offset+1)*n + 0
        w_idx_c21 = k_offset*n + 1
        w_idx_c22 = (k_offset+1)*n + 1
        
        # M1 = (a+d)(e+h)
        U[u_idx_a][col] += 1
        U[u_idx_d][col] += 1
        V[v_idx_e][col] += 1
        V[v_idx_h][col] += 1
        W[w_idx_c11][col] += 1
        W[w_idx_c22][col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[u_idx_c][col] += 1
        U[u_idx_d][col] += 1
        V[v_idx_e][col] += 1
        W[w_idx_c21][col] += 1
        col += 1
        
        # M3 = a(h-f)
        U[u_idx_a][col] += 1
        V[v_idx_h][col] += 1
        V[v_idx_f][col] -= 1
        W[w_idx_c12][col] += 1
        W[w_idx_c22][col] += 1
        col += 1
        
        # M4 = d(g-h)
        U[u_idx_d][col] += 1
        V[v_idx_g][col] += 1
        V[v_idx_h][col] -= 1
        W[w_idx_c11][col] -= 1
        W[w_idx_c12][col] += 1
        col += 1
        
        # M5 = (a+c)h
        U[u_idx_a][col] += 1
        U[u_idx_c][col] += 1
        V[v_idx_h][col] += 1
        W[w_idx_c11][col] += 1
        W[w_idx_c22][col] += 1
        col += 1
        
        # M6 = (b-d)(e+g)
        U[u_idx_b][col] += 1
        U[u_idx_d][col] -= 1
        V[v_idx_e][col] += 1
        V[v_idx_g][col] += 1
        W[w_idx_c22][col] += 1
        col += 1
        
        # M7 = (b+c)(f-h)
        U[u_idx_b][col] += 1
        U[u_idx_c][col] += 1
        V[v_idx_f][col] += 1
        V[v_idx_h][col] -= 1
        W[w_idx_c11][col] -= 1
        W[w_idx_c21][col] += 1
        col += 1
