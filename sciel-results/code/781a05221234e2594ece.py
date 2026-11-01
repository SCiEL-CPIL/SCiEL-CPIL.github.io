def add_223_block(i0, j0, k0):
    """Add a 2x2 x 2x3 schoolbook block with rank 8"""
    nonlocal col
    a_idx = i0 * m + j0
    b_idx = i0 * m + (j0 + 1)
    c_idx = (i0 + 1) * m + j0
    d_idx = (i0 + 1) * m + (j0 + 1)
    
    e_idx = j0 * p + k0
    f_idx = j0 * p + (k0 + 1)
    g_idx = j0 * p + (k0 + 2)
    h_idx = (j0 + 1) * p + k0
    i_idx = (j0 + 1) * p + (k0 + 1)
    j_idx = (j0 + 1) * p + (k0 + 2)
    
    c00_idx = k0 * n + i0
    c01_idx = (k0 + 1) * n + i0
    c02_idx = (k0 + 2) * n + i0
    c10_idx = k0 * n + (i0 + 1)
    c11_idx = (k0 + 1) * n + (i0 + 1)
    c12_idx = (k0 + 2) * n + (i0 + 1)
    
    # Row 0 of A:
    U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
    U[a_idx, col] += 1; V[f_idx, col] += 1; W[c01_idx, col] += 1; col += 1
    U[a_idx, col] += 1; V[g_idx, col] += 1; W[c02_idx, col] += 1; col += 1
    U[b_idx, col] += 1; V[h_idx, col] += 1; W[c00_idx, col] += 1; col += 1
    U[b_idx, col] += 1; V[i_idx, col] += 1; W[c01_idx, col] += 1; col += 1
    U[b_idx, col] += 1; V[j_idx, col] += 1; W[c02_idx, col] += 1; col += 1
    # Row 1 of A:
    U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
    U[c_idx, col] += 1; V[f_idx, col] += 1; W[c11_idx, col] += 1; col += 1
    U[c_idx, col] += 1; V[g_idx, col] += 1; W[c12_idx, col] += 1; col += 1
    U[d_idx, col] += 1; V[h_idx, col] += 1; W[c10_idx, col] += 1; col += 1
    U[d_idx, col] += 1; V[i_idx, col] += 1; W[c11_idx, col] += 1; col += 1
    U[d_idx, col] += 1; V[j_idx, col] += 1; W[c12_idx, col] += 1; col += 1
