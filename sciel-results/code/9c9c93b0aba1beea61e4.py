<<<<<<< SEARCH
def _get_block_decomp(ns, ms, ps):
    """Get decomposition for a block of size ns x ms x ps.
    Uses Strassen for 2x2x2, schoolbook otherwise.
    Returns (U, V, W, rank) where U has ns*ms rows, V has ms*ps rows, W has ps*ns rows.
    """
    if ns == 2 and ms == 2 and ps == 2:
        U, V, W = _strassen_2x2x2()
        return U, V, W, 7
    
    # For other blocks, try to further decompose
    # 2x2x3: split ps into 1+2 -> 2x2x1 (schoolbook rank 4) + 2x2x2 (Strassen rank 7) = 11
    if ns == 2 and ms == 2 and ps == 3:
        U1, V1, W1, r1 = _get_block_decomp(2, 2, 1)
        U2, V2, W2, r2 = _get_block_decomp(2, 2, 2)
        U = np.hstack([U1, U2])
        V = np.hstack([V1, V2])
        W = np.hstack([W1, W2])
        return U, V, W, r1 + r2
    
    # 2x2x4: split ps into 2+2 -> two 2x2x2 Strassen blocks = 14
    if ns == 2 and ms == 2 and ps == 4:
        U1, V1, W1, r1 = _get_block_decomp(2, 2, 2)
        U2, V2, W2, r2 = _get_block_decomp(2, 2, 2)
        U = np.hstack([U1, U2])
        V = np.hstack([V1, V2])
        W = np.hstack([W1, W2])
        return U, V, W, r1 + r2
    
    # 2x3x3: split ms into 1+2, ps into 1+2
    # This gets complex, use schoolbook for now
    # 2x3x4: use schoolbook for now
    
    # Schoolbook
    rank = ns * ms * ps
    U = np.zeros((ns * ms, rank))
    V = np.zeros((ms * ps, rank))
    W = np.zeros((ps * ns, rank))
    col = 0
    for i in range(ns):
        for j in range(ms):
            for k in range(ps):
                U[i * ms + j, col] = 1
                V[j * ps + k, col] = 1
                W[k * ns + i, col] = 1
                col += 1
    return U, V, W, rank
=======
def _get_block_decomp(ns, ms, ps):
    """Get decomposition for a block of size ns x ms x ps.
    Uses Strassen for 2x2x2, recursive splitting for other blocks.
    Returns (U, V, W, rank) where U has ns*ms rows, V has ms*ps rows, W has ps*ns rows.
    """
    # Base case: Strassen for 2x2x2
    if ns == 2 and ms == 2 and ps == 2:
        U, V, W = _strassen_2x2x2()
        return U, V, W, 7
    
    # Try splitting each dimension into two parts
    best_U, best_V, best_W, best_rank = None, None, None, ns * ms * ps
    
    # Try splitting ps into ps1 + ps2
    for ps1 in range(1, ps):
        ps2 = ps - ps1
        U1, V1, W1, r1 = _get_block_decomp(ns, ms, ps1)
        U2, V2, W2, r2 = _get_block_decomp(ns, ms, ps2)
        total_rank = r1 + r2
        if total_rank < best_rank:
            U = np.hstack([U1, U2])
            V = np.hstack([V1, V2])
            W = np.hstack([W1, W2])
            best_U, best_V, best_W, best_rank = U, V, W, total_rank
    
    # Try splitting ms into ms1 + ms2
    for ms1 in range(1, ms):
        ms2 = ms - ms1
        U1, V1, W1, r1 = _get_block_decomp(ns, ms1, ps)
        U2, V2, W2, r2 = _get_block_decomp(ns, ms2, ps)
        total_rank = r1 + r2
        if total_rank < best_rank:
            U = np.hstack([U1, U2])
            V = np.hstack([V1, V2])
            W = np.hstack([W1, W2])
            best_U, best_V, best_W, best_rank = U, V, W, total_rank
    
    # Try splitting ns into ns1 + ns2
    for ns1 in range(1, ns):
        ns2 = ns - ns1
        U1, V1, W1, r1 = _get_block_decomp(ns1, ms, ps)
        U2, V2, W2, r2 = _get_block_decomp(ns2, ms, ps)
        total_rank = r1 + r2
        if total_rank < best_rank:
            U = np.hstack([U1, U2])
            V = np.hstack([V1, V2])
            W = np.hstack([W1, W2])
            best_U, best_V, best_W, best_rank = U, V, W, total_rank
    
    if best_U is not None:
        return best_U, best_V, best_W, best_rank
    
    # Schoolbook
    rank = ns * ms * ps
    U = np.zeros((ns * ms, rank))
    V = np.zeros((ms * ps, rank))
    W = np.zeros((ps * ns, rank))
    col = 0
    for i in range(ns):
        for j in range(ms):
            for k in range(ps):
                U[i * ms + j, col] = 1
                V[j * ps + k, col] = 1
                W[k * ns + i, col] = 1
                col += 1
    return U, V, W, rank
>>>>>>> REPLACE

Now let me also fix the verification function to be more efficient and correct. The current verification is O(n²·m²·p²·rank) which is too slow. Let me use a smarter check.

Actually, for a 3×5×7 problem with rank ~100, the verification should be fast enough if I use NumPy vectorization. Let me rewrite it.

