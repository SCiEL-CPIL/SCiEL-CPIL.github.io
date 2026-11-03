import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n*m, m*p, p*n
    
    all_terms = set((i,j,k) for i in range(n) for j in range(m) for k in range(p))
    
    # Generate all 2x2x2 blocks
    blocks = []
    for i1 in range(n):
        for i2 in range(i1+1, n):
            for j1 in range(m):
                for j2 in range(j1+1, m):
                    for k1 in range(p):
                        for k2 in range(k1+1, p):
                            terms = frozenset((i,j,k) for i in (i1,i2) for j in (j1,j2) for k in (k1,k2))
                            blocks.append(((i1,i2,j1,j2,k1,k2), terms))
    
    # Greedy: pick block with max new terms
    covered = set()
    selected = []
    while True:
        if time.time() > deadline - 5:
            break
        best_gain, best_block = 0, None
        for params, terms in blocks:
            gain = len(terms - covered)
            if gain > best_gain:
                best_gain, best_block = gain, params
        if best_gain <= 0:
            break
        for params, terms in blocks:
            if params == best_block:
                covered |= terms
                break
        selected.append(best_block)
    
    # Build decomposition
    U_cols, V_cols, W_cols = [], [], []
    
    def add_term(u_e, v_e, w_e):
        U_cols.append([0]*U_rows); V_cols.append([0]*V_rows); W_cols.append([0]*W_rows)
        for idx, val in u_e.items(): U_cols[-1][idx] = val
        for idx, val in v_e.items(): V_cols[-1][idx] = val
        for idx, val in w_e.items(): W_cols[-1][idx] = val
    
    def strassen(i1,i2,j1,j2,k1,k2):
        a11,a12,a21,a22 = i1*m+j1, i1*m+j2, i2*m+j1, i2*m+j2
        b11,b12,b21,b22 = j1*p+k1, j1*p+k2, j2*p+k1, j2*p+k2
        c11,c12,c21,c22 = k1*n+i1, k2*n+i1, k1*n+i2, k2*n+i2
        add_term({a11:1,a22:1},{b11:1,b22:1},{c11:1,c22:1})
        add_term({a21:1,a22:1},{b11:1},{c21:1,c22:-1})
        add_term({a11:1},{b12:1,b22:-1},{c12:1,c22:1})
        add_term({a22:1},{b21:1,b11:-1},{c11:1,c21:1})
        add_term({a11:1,a12:1},{b22:1},{c11:-1,c12:1})
        add_term({a21:1,a11:-1},{b11:1,b12:1},{c22:1})
        add_term({a12:1,a22:-1},{b21:1,b22:1},{c11:1})
    
    for block in selected:
        strassen(*block)
    
    for (i,j,k) in all_terms:
        if (i,j,k) not in covered:
            add_term({i*m+j:1},{j*p+k:1},{k*n+i:1})
    
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(U_rows)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(V_rows)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(W_rows)]
    return {"U": U, "V": V, "W": W}
