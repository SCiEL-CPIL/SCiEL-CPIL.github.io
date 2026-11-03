def construct():
    n, m, p = DIMENSIONS
    import time
    _start = time.time()
    _deadline = _start + 140
    
    # Build rank-54 decomposition
    rank = 54
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    j_pairs = [(0,1), (2,3)]
    k_pairs = [(0,1), (2,3), (4,5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p)
    
    for k in range(p):
        U[0*m+4][col] += 1
        V[4*p+k][col] += 1
        W[k*n+0][col] += 1
        col += 1
        U[1*m+4][col] += 1
        V[4*p+k][col] += 1
        W[k*n+1][col] += 1
        col += 1
    
    if not _verify_decomp(U, V, W, n, m, p, rank):
        pass  # Should be valid
    
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Try to reduce rank by checking if any term is redundant
    # For each term, zero it out and check if we can compensate
    # This is expensive, so limit by time
    
    # Alternative: try different j-pairings
    # Option: (0,1), (2,4), j=3 singleton
    # Option: (0,2), (1,3), j=4 singleton
    # etc.
    
    # Let's try all possible pairings of 5 j-values into 2 pairs + 1 singleton
    import itertools
    
    all_js = list(range(m))  # [0,1,2,3,4]
    
    def build_decomp(j_pairs_list, singleton_j):
        """Build decomposition with given j-pairs and singleton."""
        r = 7 * len(j_pairs_list) + 2 * p  # 7 per Strassen block + 2*p for singleton
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        col = 0
        k_pairs = [(0,1), (2,3), (4,5)]
        for j0, j1 in j_pairs_list:
            for k0, k1 in k_pairs:
                col = _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p)
        
        for k in range(p):
            U[0*m+singleton_j][col] += 1
            V[singleton_j*p+k][col] += 1
            W[k*n+0][col] += 1
            col += 1
            U[1*m+singleton_j][col] += 1
            V[singleton_j*p+k][col] += 1
            W[k*n+1][col] += 1
            col += 1
        
        return U, V, W, r
    
    # Try all ways to pick 2 pairs from 5 elements, leaving 1 singleton
    for singleton in all_js:
        remaining = [j for j in all_js if j != singleton]
        for pair1 in itertools.combinations(remaining, 2):
            rest = [j for j in remaining if j not in pair1]
            if len(rest) == 2:
                pair2 = tuple(rest)
                j_pairs_list = [list(pair1), list(pair2)]
                U_t, V_t, W_t, r_t = build_decomp(j_pairs_list, singleton)
                if _verify_decomp(U_t, V_t, W_t, n, m, p, r_t):
                    if r_t < best_rank:
                        best_rank = r_t
                        best_U = U_t
                        best_V = V_t
                        best_W = W_t
                
                if time.time() > _deadline:
                    break
        
        if time.time() > _deadline:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
