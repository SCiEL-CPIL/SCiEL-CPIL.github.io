def add_strassen_block(U, V, W, offset, a_idx, b_idx, c_idx, d_idx, e_idx, f_idx, g_idx, h_idx, w_11, w_12, w_21, w_22, sign=1):
    # a,b,c,d are indices in U
    # e,f,g,h are indices in V
    # w_11, w_12, w_21, w_22 are indices in W for C_11, C_12, C_21, C_22
    
    # M1 = (a+d)(e+h)
    col = offset + 0
    U[a_idx][col] += sign
    U[d_idx][col] += sign
    V[e_idx][col] += sign
    V[h_idx][col] += sign
    W[w_11][col] += sign
    W[w_22][col] += sign

    # M2 = (c+d)e
    col = offset + 1
    U[c_idx][col] += sign
    U[d_idx][col] += sign
    V[e_idx][col] += sign
    W[w_21][col] += sign
    W[w_22][col] -= sign

    # M3 = a(f-h)
    col = offset + 2
    U[a_idx][col] += sign
    V[f_idx][col] += sign
    V[h_idx][col] -= sign
    W[w_12][col] += sign
    W[w_22][col] += sign

    # M4 = d(g-e)
    col = offset + 3
    U[d_idx][col] += sign
    V[g_idx][col] += sign
    V[e_idx][col] -= sign
    W[w_11][col] += sign
    W[w_21][col] += sign

    # M5 = (a+b)h
    col = offset + 4
    U[a_idx][col] += sign
    U[b_idx][col] += sign
    V[h_idx][col] += sign
    W[w_11][col] -= sign
    W[w_12][col] += sign

    # M6 = (c-a)(e+f)
    col = offset + 5
    U[c_idx][col] += sign
    U[a_idx][col] -= sign
    V[e_idx][col] += sign
    V[f_idx][col] += sign
    W[w_22][col] += sign

    # M7 = (b-d)(g+h)
    col = offset + 6
    U[b_idx][col] += sign
    U[d_idx][col] -= sign
    V[g_idx][col] += sign
    V[h_idx][col] += sign
    W[w_11][col] += sign
