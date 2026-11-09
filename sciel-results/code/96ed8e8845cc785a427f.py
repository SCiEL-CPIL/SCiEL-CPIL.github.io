"""Combined program: schoolbook fallback + Strassen-based block decomposition for 3x4x7 matrix multiplication."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = start_time + 115  # 120s limit, keep 5s margin

    n, m, p = DIMENSIONS  # 3, 4, 7

    def schoolbook():
        """Standard schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()

    def verify(U, V, W, n, m, p):
        """Verify exactness: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    # Precompute Strassen 2x2x2 decomposition (computed once)
    _strassen_222_cache = None
    def strassen_222():
        """
        Strassen decomposition for 2x2x2 matrix multiplication.
        Returns (U, V, W) where:
          U: shape (4, 7) - coefficients for A entries
          V: shape (4, 7) - coefficients for B entries
          W: shape (4, 7) - coefficients for C entries
        Indexing: U[i*2+j, c] for A_{i,j}, V[j*2+k, c] for B_{j,k}, W[k*2+i, c] for C_{i,k}
        """
        nonlocal _strassen_222_cache
        if _strassen_222_cache is not None:
            return _strassen_222_cache
        
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]

        # W indexing: W[k*n + i, c] where C[i,k], n=2
        # C[0,0] -> W[0], C[0,1] -> W[2], C[1,0] -> W[1], C[1,1] -> W[3]

        # M1: (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1   # A00, A11
        V[0][0] = 1; V[3][0] = 1   # B00, B11
        W[0][0] = 1                # C00 += M1

        # M2: (A10+A11)(B00)
        U[2][1] = 1; U[3][1] = 1   # A10, A11
        V[0][1] = 1                # B00
        W[1][1] = 1                # C10 += M2

        # M3: A00(B01-B11)
        U[0][2] = 1                # A00
        V[1][2] = 1; V[3][2] = -1  # B01 - B11
        W[2][2] = 1                # C01 += M3

        # M4: A11(B10-B00)
        U[3][3] = 1                # A11
        V[2][3] = 1; V[0][3] = -1  # B10 - B00
        W[0][3] = 1                # C00 += M4
        W[1][3] = 1                # C10 += M4

        # M5: (A00+A01)(B11)
        U[0][4] = 1; U[1][4] = 1   # A00, A01
        V[3][4] = 1                # B11
        W[2][4] = 1                # C01 += M5
        W[0][4] = -1               # C00 -= M5

        # M6: (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1  # A10 - A00
        V[0][5] = 1; V[1][5] = 1   # B00 + B01
        W[3][5] = 1                # C11 += M6

        # M7: (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1  # A01 - A11
        V[2][6] = 1; V[3][6] = 1   # B10 + B11
        W[0][6] = 1                # C00 += M7
        W[3][6] = 1                # C11 += M7

        _strassen_222_cache = (U, V, W)
        return U, V, W



    def try_block_decompose(n_splits, m_splits, p_splits):
        """Try a specific block decomposition and return the result if valid."""
        n, m, p = 3, 4, 7
        total_rank = 0
        # Calculate expected rank
        for i_sub in n_splits:
            for j_sub in m_splits:
                for k_sub in p_splits:
                    if i_sub == 2 and j_sub == 2 and k_sub == 2:
                        total_rank += 7  # Strassen
                    else:
                        total_rank += i_sub * j_sub * k_sub  # Schoolbook
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        i_offsets = [0]
        for ns in n_splits[:-1]:
            i_offsets.append(i_offsets[-1] + ns)
        j_offsets = [0]
        for ms in m_splits[:-1]:
            j_offsets.append(j_offsets[-1] + ms)
        k_offsets = [0]
        for ps in p_splits[:-1]:
            k_offsets.append(k_offsets[-1] + ps)
        
        for i_idx, i_sub in enumerate(n_splits):
            for j_idx, j_sub in enumerate(m_splits):
                for k_idx, k_sub in enumerate(p_splits):
                    i_off = i_offsets[i_idx]
                    j_off = j_offsets[j_idx]
                    k_off = k_offsets[k_idx]
                    if i_sub == 2 and j_sub == 2 and k_sub == 2:
                        col = add_strassen_block(U, V, W, i_off, j_off, k_off, col)
                    else:
                        col = add_schoolbook_block(U, V, W, i_off, j_off, k_off, col, i_sub, j_sub, k_sub)
        
        if col != total_rank:
            return None
        
        if verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        return None
    
    # Try the original block decomposition and alternatives
    p_splits_to_try = [
        [2, 2, 2, 1],
        [2, 2, 1, 2],
        [2, 1, 2, 2],
        [1, 2, 2, 2],
    ]
    
    for ps in p_splits_to_try:
        if time.time() > DEADLINE - 5:
            break
        try:
            result = try_block_decompose([2, 1], [2, 2], ps)
            if result is not None and len(result["U"][0]) < len(best["U"][0]):
                best = result
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
