def _try_merge_adjacent_blocks(n, m, p, deadline):
    """
    Attempts to merge the 1x2x2, 2x1x2, and 2x2x1 blocks adjacent to the 
    2x2x2 Strassen core into the core's columns.
    Returns U, V, W if successful, else None.
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # We will try to solve for the "extension" factors for each adjacent block.
    # Block 1: 1x2x2 (i=2, j=0,1, k=0,1). Shares j,k.
    # Needs X1 (2x7) for U[2, j, r] and Y1 (2x7) for W[k, 2, r].
    X1, Y1 = _solve_merge_1x2x2(V_s)
    if X1 is None: return None
    
    # Block 2: 2x1x2 (i=0,1, j=2, k=0,1). Shares i,k.
    # Needs X2 (2x7) for U[0..1, 2, r] and Y2 (2x7) for V[2, k, r]?
    # Eq: sum_r U[i, 2, r] V[2, k, r] W[k, i, r] = 1.
    # W is fixed (core). U and V are free.
    X2, Y2 = _solve_merge_2x1x2(W_s)
    if X2 is None: return None
    
    # Block 3: 2x2x1 (i=0,1, j=0,1, k=2). Shares i,j.
    # Needs X3 (2x7) for U[0..1, 0..1, r]? No, U is fixed.
    # Eq: sum_r U[i, j, r] V[j, 2, r] W[2, i, r] = 1.
    # U is fixed. V and W are free.
    X3, Y3 = _solve_merge_2x2x1(U_s)
    if X3 is None: return None
    
    # Construct full U, V, W
    # Rank = 7 (core) + Schoolbook for the rest.
    # Rest blocks:
    # 1x2x3 (i=2, j=0,1, k=2,3,4) -> Rank 6
    # 1x3x2 (i=2, j=2,3,4, k=0,1) -> Rank 6
    # 1x3x3 (i=2, j=2,3,4, k=2,3,4) -> Rank 9
    # 2x3x2 (i=0,1, j=2,3,4, k=0,1) -> Rank 12
    # 2x2x3 (i=0,1, j=0,1, k=2,3,4) -> Rank 12
    # 2x3x3 (i=0,1, j=2,3,4, k=2,3,4) -> Rank 18
    # Total Schoolbook Rank = 6+6+9+12+12+18 = 63.
    # Total Rank = 7 + 63 = 70.
    
    total_rank = 70
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Fill Core (cols 0-6)
    # Embed Strassen
    U_core_g, V_core_g, W_core_g = _embed_block(U_s, V_s, W_s, n, m, p, 0, 0, 0, 2, 2, 2)
    U[:, 0:7] = U_core_g[:, 0:7]
    V[:, 0:7] = V_core_g[:, 0:7]
    W[:, 0:7] = W_core_g[:, 0:7]
    
    # Add Merged Block 1 (1x2x2)
    # U[2, j, r] = X1[j, r]
    for j in range(2):
        U[2*m + j, 0:7] = X1[j, :]
    # W[k, 2, r] = Y1[k, r]
    for k in range(2):
        W[k*n + 2, 0:7] = Y1[k, :]
        
    # Add Merged Block 2 (2x1x2)
    # U[i, 2, r] = X2[i, r]
    for i in range(2):
        U[i*m + 2, 0:7] = X2[i, :]
    # V[2, k, r] = Y2[k, r]
    for k in range(2):
        V[2*p + k, 0:7] = Y2[k, :]
        
    # Add Merged Block 3 (2x2x1)
    # V[j, 2, r] = X3[j, r]
    for j in range(2):
        V[j*p + 2, 0:7] = X3[j, :]
    # W[2, i, r] = Y3[i, r]
    for i in range(2):
        W[2*n + i, 0:7] = Y3[i, :]
        
    # Fill Schoolbook Blocks
    col = 7
    # Define schoolbook blocks
    blocks = [
        (2, 0, 2, 1, 2, 3), # i=2, j=0,1, k=2,3,4
        (2, 2, 0, 1, 3, 2), # i=2, j=2,3,4, k=0,1
        (2, 2, 2, 1, 3, 3), # i=2, j=2,3,4, k=2,3,4
        (0, 2, 0, 2, 3, 2), # i=0,1, j=2,3,4, k=0,1
        (0, 0, 2, 2, 2, 3), # i=0,1, j=0,1, k=2,3,4
        (0, 2, 2, 2, 3, 3), # i=0,1, j=2,3,4, k=2,3,4
    ]
    
    for i_off, j_off, k_off, ni, mi, pi in blocks:
        U_loc, V_loc, W_loc = _schoolbook_local(ni, mi, pi)
        r = ni * mi * pi
        U_g, V_g, W_g = _embed_block(U_loc, V_loc, W_loc, n, m, p, i_off, j_off, k_off, ni, mi, pi)
        U[:, col:col+r] += U_g
        V[:, col:col+r] += V_g
        W[:, col:col+r] += W_g
        col += r
        
    # Verify
    if _verify(U, V, W, n, m, p):
        return U, V, W, total_rank
        
    return None
