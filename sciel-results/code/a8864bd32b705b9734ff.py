def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # We'll use a recursive decomposition
    # First, compute the total rank needed
    def compute_rank(ni, mj, pk):
        if ni == 2 and mj == 2 and pk == 2:
            return 7
        if ni <= 2 and mj <= 2 and pk <= 2:
            return ni * mj * pk
        # Split largest dimension
        if ni >= mj and ni >= pk and ni > 2:
            return compute_rank(2, mj, pk) + compute_rank(ni-2, mj, pk)
        elif mj >= ni and mj >= pk and mj > 2:
            return compute_rank(ni, 2, pk) + compute_rank(ni, mj-2, pk)
        elif pk >= ni and pk >= mj and pk > 2:
            return compute_rank(ni, mj, 2) + compute_rank(ni, mj, pk-2)
        else:
            # All <= 2 but not all 2
            return ni * mj * pk
    
    total_rank = compute_rank(n, m, p)
    
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    # Strassen terms for 2x2x2
    strassen_terms = [
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,1],[0,-1]]),
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,0],[1,1]]),
        ([[0,0],[0,1]], [[-1,0],[1,0]], [[1,1],[0,0]]),
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[1,0]]),
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]
    
    def add_strassen(U, V, W, col, i_off, j_off, k_off):
        for u_mat, v_mat, w_mat in strassen_terms:
            for ii in range(2):
                for jj in range(2):
                    if u_mat[ii][jj] != 0:
                        U[(i_off+ii)*m + (j_off+jj), col] += u_mat[ii][jj]
            for jj in range(2):
                for kk in range(2):
                    if v_mat[jj][kk] != 0:
                        V[(j_off+jj)*p + (k_off+kk), col] += v_mat[jj][kk]
            for kk in range(2):
                for ii in range(2):
                    if w_mat[kk][ii] != 0:
                        W[(k_off+kk)*n + (i_off+ii), col] += w_mat[kk][ii]
            col += 1
        return col
    
    def add_schoolbook(U, V, W, col, i_off, j_off, k_off, ni, mj, pk):
        for ii in range(ni):
            for jj in range(mj):
                for kk in range(pk):
                    U[(i_off+ii)*m + (j_off+jj), col] = 1.0
                    V[(j_off+jj)*p + (k_off+kk), col] = 1.0
                    W[(k_off+kk)*n + (i_off+ii), col] = 1.0
                    col += 1
        return col
    
    def decompose(ni, mj, pk, i_off, j_off, k_off, U, V, W, col):
        if time.time() > deadline:
            raise TimeoutError()
        if ni == 2 and mj == 2 and pk == 2:
            return add_strassen(U, V, W, col, i_off, j_off, k_off)
        if ni <= 2 and mj <= 2 and pk <= 2:
            return add_schoolbook(U, V, W, col, i_off, j_off, k_off, ni, mj, pk)
        # Split largest dimension
        if ni >= mj and ni >= pk and ni > 2:
            col = decompose(2, mj, pk, i_off, j_off, k_off, U, V, W, col)
            col = decompose(ni-2, mj, pk, i_off+2, j_off, k_off, U, V, W, col)
        elif mj >= ni and mj >= pk and mj > 2:
            col = decompose(ni, 2, pk, i_off, j_off, k_off, U, V, W, col)
            col = decompose(ni, mj-2, pk, i_off, j_off+2, k_off, U, V, W, col)
        elif pk >= ni and pk >= mj and pk > 2:
            col = decompose(ni, mj, 2, i_off, j_off, k_off, U, V, W, col)
            col = decompose(ni, mj, pk-2, i_off, j_off, k_off+2, U, V, W, col)
        else:
            return add_schoolbook(U, V, W, col, i_off, j_off, k_off, ni, mj, pk)
        return col
    
    col = decompose(n, m, p, 0, 0, 0, U, V, W, 0)
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    # Verify
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(total_rank))
                if abs(val - 1.0) > 1e-9:
                    error += abs(val - 1.0)
    
    if error > 1e-6:
        rank_school = n * m * p
        U_s = np.zeros((nm, rank_school), dtype=float)
        V_s = np.zeros((mp, rank_school), dtype=float)
        W_s = np.zeros((pn, rank_school), dtype=float)
        col2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, col2] = 1.0
                    V_s[j*p+k, col2] = 1.0
                    W_s[k*n+i, col2] = 1.0
                    col2 += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
