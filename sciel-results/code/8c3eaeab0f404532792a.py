def add_strassen_2x2(a_idx, b_idx, c_idx, start_col, n, m, p):
    # a_idx: list of (i, j)
    # b_idx: list of (j, k)
    # c_idx: list of (i, k)
    
    # Term 1
    r = start_col
    i0, j0 = a_idx[0]
    i3, j3 = a_idx[3]
    j0b, k0b = b_idx[0]
    j3b, k3b = b_idx[3]
    i0c, k0c = c_idx[0]
    i3c, k3c = c_idx[3]
    
    U[i0*m + j0][r] += 1.0
    U[i3*m + j3][r] += 1.0
    V[j0b*p + k0b][r] += 1.0
    V[j3b*p + k3b][r] += 1.0
    W[k0c*n + i0c][r] += 1.0
    W[k3c*n + i3c][r] += 1.0
    # ... and so on
