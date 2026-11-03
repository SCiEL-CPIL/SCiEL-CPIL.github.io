def _add_strassen_block(U, V, W, n, m, p, a_rows, a_cols, b_rows, b_cols, col):
    """
    Add 7 rank-1 terms for the 2x2 Strassen block.
    a_rows, a_cols, b_rows, b_cols are lists of 2 indices.
    """
    a0, a1 = a_rows
    j0, j1 = a_cols
    k0, k1 = b_rows
    c0, c1 = b_cols
    
    # Define local index mapping to global indices
    # A indices: (a0, j0), (a0, j1), (a1, j0), (a1, j1)
    # B indices: (j0, k0), (j0, k1), (j1, k0), (j1, k1)
    # C indices: (k0, a0), (k0, a1), (k1, a0), (k1, a1)
    
    def idx_A(r, c): return r * m + c
    def idx_B(r, c): return r * p + c
    def idx_C(r, c): return r * n + c
    
    # m1 = (a00+a11)(b00+b11)
    # u: a00=1, a11=1
    # v: b00=1, b11=1
    # w: c00=1, c11=1
    U[idx_A(a0, j0), col] = 1
    U[idx_A(a1, j1), col] = 1
    V[idx_B(j0, k0), col] = 1
    V[idx_B(j1, k1), col] = 1
    W[idx_C(k0, a0), col] = 1
    W[idx_C(k1, a1), col] = 1
    col += 1
    
    # m2 = (a10+a11)b00
    # u: a10=1, a11=1
    # v: b00=1
    # w: c10=1, c11=-1
    U[idx_A(a1, j0), col] = 1
    U[idx_A(a1, j1), col] = 1
    V[idx_B(j0, k0), col] = 1
    W[idx_C(k0, a1), col] = 1
    W[idx_C(k1, a1), col] = -1
    col += 1
    
    # m3 = a00(b10-b11)
    # u: a00=1
    # v: b10=1, b11=-1
    # w: c00=1, c01=1
    U[idx_A(a0, j0), col] = 1
    V[idx_B(j1, k0), col] = 1
    V[idx_B(j1, k1), col] = -1
    W[idx_C(k0, a0), col] = 1
    W[idx_C(k1, a0), col] = 1
    col += 1
    
    # m4 = a11(b01-b00)
    # u: a11=1
    # v: b01=1, b00=-1
    # w: c10=1, c11=1
    U[idx_A(a1, j1), col] = 1
    V[idx_B(j0, k1), col] = 1
    V[idx_B(j0, k0), col] = -1
    W[idx_C(k0, a1), col] = 1
    W[idx_C(k1, a1), col] = 1
    col += 1
    
    # m5 = (a00+a01)b11
    # u: a00=1, a01=1
    # v: b11=1
    # w: c01=1, c11=-1
    U[idx_A(a0, j0), col] = 1
    U[idx_A(a0, j1), col] = 1
    V[idx_B(j1, k1), col] = 1
    W[idx_C(k1, a0), col] = 1
    W[idx_C(k1, a1), col] = -1
    col += 1
    
    # m6 = (a10-a00)(b00+b10)
    # u: a10=1, a00=-1
    # v: b00=1, b10=1
    # w: c10=1, c11=1
    U[idx_A(a1, j0), col] = 1
    U[idx_A(a0, j0), col] = -1
    V[idx_B(j0, k0), col] = 1
    V[idx_B(j1, k0), col] = 1
    W[idx_C(k0, a1), col] = 1
    W[idx_C(k1, a1), col] = 1
    col += 1
    
    # m7 = (a01-a11)(b01+b11)
    # u: a01=1, a11=-1
    # v: b01=1, b11=1
    # w: c01=1, c11=1
    U[idx_A(a0, j1), col] = 1
    U[idx_A(a1, j1), col] = -1
    V[idx_B(j0, k1), col] = 1
    V[idx_B(j1, k1), col] = 1
    W[idx_C(k1, a0), col] = 1
    W[idx_C(k1, a1), col] = 1
    col += 1
    
    return col
