<<<<<<< SEARCH
def _build_strassen_3x5x6(n, m, p):
    """
    Build tensor decomposition for 3x5 @ 5x6 using Strassen block structure.
    """
    sU, sV, sW = _get_strassen_2x2x2()
    
    # Total rank: 3 k-blocks * (2 Strassen blocks + 1 schoolbook for j=4)
    # Per k-block: 18 (top) + 10 (bottom) = 28
    # 3 k-blocks: 84
    
    rank = 84
    U = np.zeros((n * m, rank))  # 15 x 84
    V = np.zeros((m * p, rank))  # 30 x 84
    W = np.zeros((p * n, rank))  # 18 x 84
    
    col = 0
    
    for k_block in range(3):
        k0 = k_block * 2  # start k index
        k1 = k0 + 1       # end k index
        
        # Top 2 rows (i=0,1), j in {0,1}: Strassen
        for c in range(7):
            U[0*5 + 0, col] = sU[0, c]
            U[0*5 + 1, col] = sU[1, c]
            U[1*5 + 0, col] = sU[2, c]
            U[1*5 + 1, col] = sU[3, c]
            V[0*6 + k0, col] = sV[0, c]
            V[0*6 + k1, col] = sV[1, c]
            V[1*6 + k0, col] = sV[2, c]
            V[1*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
        
        # Top 2 rows (i=0,1), j in {2,3}: Strassen
        for c in range(7):
            U[0*5 + 2, col] = sU[0, c]
            U[0*5 + 3, col] = sU[1, c]
            U[1*5 + 2, col] = sU[2, c]
            U[1*5 + 3, col] = sU[3, c]
            V[2*6 + k0, col] = sV[0, c]
            V[2*6 + k1, col] = sV[1, c]
            V[3*6 + k0, col] = sV[2, c]
            V[3*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
        
        # Top 2 rows (i=0,1), j=4: schoolbook 2x1x2
        for i in range(2):
            for k in range(2):
                U[i*5 + 4, col] = 1
                V[4*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + i, col] = 1
                col += 1
        
        # Bottom row (i=2), j in {0,1}: schoolbook 1x2x2
        for j in range(2):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        
        # Bottom row (i=2), j in {2,3}: schoolbook 1x2x2
        for j in range(2, 4):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        
        # Bottom row (i=2), j=4: schoolbook 1x1x2
        for k in range(2):
            U[2*5 + 4, col] = 1
            V[4*6 + k0 + k, col] = 1
            W[(k0 + k)*3 + 2, col] = 1
            col += 1
    
    if col != rank:
        # Truncate or pad
        if col > rank:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
        else:
            rank = col
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
    
    return U, V, W, rank
=======
def _build_strassen_3x5x6(n, m, p):
    """
    Build tensor decomposition for 3x5 @ 5x6 by projecting a 6x6x6
    Strassen decomposition (rank 63) onto the 3x5x6 subspace.
    """
    sU, sV, sW = _get_strassen_2x2x2()
    
    # Build full 6x6x6 Strassen decomposition: rank = 3*3*7 = 63
    full_rank = 63
    full_U = np.zeros((36, full_rank))  # 6*6 = 36
    full_V = np.zeros((36, full_rank))  # 6*6 = 36
    full_W = np.zeros((36, full_rank))  # 6*6 = 36
    
    col = 0
    for i_block in range(3):
        for j_block in range(3):
            for c in range(7):
                # U: A[i_block*2 + di, j_block*2 + dj]
                for di in range(2):
                    for dj in range(2):
                        full_U[(i_block*2 + di)*6 + (j_block*2 + dj), col] = sU[di*2 + dj, c]
                # V: B[j_block*2 + dj, k_block*2 + dk]
                for dj in range(2):
                    for dk in range(2):
                        full_V[(j_block*2 + dj)*6 + (k_block*2 + dk), col] = sV[dj*2 + dk, c]
                # W: C[k_block*2 + dk, i_block*2 + di]
                for dk in range(2):
                    for di in range(2):
                        full_W[(k_block*2 + dk)*6 + (i_block*2 + di), col] = sW[dk*2 + di, c]
                col += 1
    
    # Project to 3x5x6:
    # U: rows (i,j) for i in 0..2, j in 0..4 -> 15 rows
    # V: rows (j,k) for j in 0..4, k in 0..5 -> 30 rows
    # W: rows (k,i) for k in 0..5, i in 0..2 -> 18 rows
    
    U_rows = []
    for i in range(n):
        for j in range(m):
            U_rows.append(i * 6 + j)
    
    V_rows = []
    for j in range(m):
        for k in range(p):
            V_rows.append(j * 6 + k)
    
    W_rows = []
    for k in range(p):
        for i in range(n):
            W_rows.append(k * 6 + i)
    
    U = full_U[U_rows, :]
    V = full_V[V_rows, :]
    W = full_W[W_rows, :]
    
    # Remove any all-zero columns (shouldn't happen, but safety)
    nonzero_cols = np.any(np.abs(U) > 1e-15, axis=0) & \
                   np.any(np.abs(V) > 1e-15, axis=0) & \
                   np.any(np.abs(W) > 1e-15, axis=0)
    U = U[:, nonzero_cols]
    V = V[:, nonzero_cols]
    W = W[:, nonzero_cols]
    
    rank = U.shape[1]
    return U, V, W, rank
>>>>>>> REPLACE

Now I need to fix the `_verify_fast` function since the factor shapes are different (U is 15×rank, V is 30×rank, W is 18×rank), and the reshape logic needs to match the correct index conventions:

