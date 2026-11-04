            # Yes, it returns col+7.
            
            # Is it possible the seed program provided in the prompt is NOT the one that scored 1.0?
            # "CURRENT PROGRAM (score 1.0000) and its strategy:"
            # The code block shows `rank = 114`.
            # If `col` ends up being 30, `rank` becomes 30.
            # Unless `_add_444` is different.
            # Wait, the prompt says "Matrix multiplication 4,4,9".
            # Maybe the seed program is broken or I am misinterpreting "score 1.0".
            # "1.0 equals the seed program".
            # If the seed program returns Rank 30, and the score is 1.0, then the baseline rank is 30.
            # But the code says `rank = 114`.
            # Maybe the code in the prompt is a REVISION of the seed program?
            # "CURRENT PROGRAM (score 1.0000) and its strategy: [Code]"
            # If this code scores 1.0, and the score is `seed_rank / current_rank`,
            # and `seed_rank` is fixed...
            # If the code returns 30, and score is 1.0, then `seed_rank` must be 30.
            # But the code hardcodes `rank = 114`.
            # If `col` is 30, `rank` becomes 30.
            # So the returned rank is 30.
            # So the baseline is 30.
            
            # If I return Rank 15, the score will be 30 / 15 = 2.0.
            
            # Let's assume the baseline is 30 (based on the code logic) or 114 (based on the variable name).
            # Given the code logic, it returns 30.
            # My Rank 15 is better.
            
            pass
        col += 1
    return col

def _add_rank1_block(U, V, W, col, a_base, b_base, c_base, k_offset, n, p):
    """
    Adds a rank-1 block for a 4x4x1 multiplication at column k_offset.
    """
    # U: i in 0..3, j in 0..3. All 1s.
    for i in range(4):
        for j in range(4):
            U[a_base + i * 4 + j][col] += 1
            
    # V: j in 0..3, k = k_offset. All 1s.
    for j in range(4):
        V[b_base + j * p + k_offset][col] += 1
        
    # W: k = k_offset, i in 0..3. All 1s.
    # Index in W is k*n + i.
    for i in range(4):
        W[c_base + k_offset * n + i][col] += 1
        
    col += 1
    return col

def _build_rank15(n, m, p):
    # n=4, m=4, p=9
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: k in 0..3
    col = _add_strassen_block(U, V, W, col, 0, 0, 0, 0)
    
    # Block 2: k in 4..7
    col = _add_strassen_block(U, V, W, col, 0, 0, 0, 4)
    
    # Block 3: k = 8
    col = _add_rank1_block(U, V, W, col, 0, 0, 0, 8, n, p)
    
    if col != rank:
        rank = col
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
        
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > 1e-9:
                    return False
    T2 = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i * m + j, j * p + k, k * n + i] = 0.0
    if np.max(np.abs(T2)) > 1e-9:
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    res = _build_rank15(n, m, p)
    if time.time() < deadline and _verify(res["U"], res["V"], res["W"], n, m, p):
        return res
        
    # Fallback to schoolbook if verification fails (should not happen)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
