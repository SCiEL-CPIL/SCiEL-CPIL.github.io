import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses 6 Strassen 2x2x2 blocks (rank 7 each) plus schoolbook for the rest.
    Total rank: 6*7 + (105 - 6*8) = 42 + 57 = 99.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries, v_entries, w_entries are dicts: {index: value}"""
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block.
        
        A submatrix: rows i1,i2, cols j1,j2
        B submatrix: rows j1,j2, cols k1,k2
        C submatrix: rows i1,i2, cols k1,k2
        
        U indices: i*m+j
        V indices: j*p+k
        W indices: k*n+i
        """
        # A positions
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        # B positions
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        # C positions
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (a11+a22)*(b11+b22) -> W: +1 at c11, +1 at c22
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11 -> W: +1 at c21, -1 at c22
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22) -> W: +1 at c12, +1 at c22
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11) -> W: +1 at c11, +1 at c21
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22 -> W: -1 at c11, +1 at c12
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12) -> W: +1 at c22
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22) -> W: +1 at c11
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    # Try multiple block configurations and keep the one with lowest rank
    best_rank = float('inf')
    best_U_cols = None
    best_V_cols = None
    best_W_cols = None

    def try_configuration(blocks):
        """Try a configuration of Strassen blocks, return (rank, U_cols, V_cols, W_cols)."""
        nonlocal U_cols, V_cols, W_cols
        U_cols = []
        V_cols = []
        W_cols = []
        covered = set()
        for (i1, i2, j1, j2, k1, k2) in blocks:
            add_strassen_block(i1, i2, j1, j2, k1, k2)
            for i in (i1, i2):
                for j in (j1, j2):
                    for k in (k1, k2):
                        covered.add((i, j, k))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})
        return len(U_cols), U_cols[:], V_cols[:], W_cols[:]

    # Configuration 1: original (i in {0,1})
    config1 = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]

    # Configuration 2: add blocks covering i=2 (using i in {1,2})
    config2 = config1 + [
        (1, 2, 0, 1, 0, 1),
        (1, 2, 0, 1, 2, 3),
        (1, 2, 0, 1, 4, 5),
        (1, 2, 2, 3, 0, 1),
        (1, 2, 2, 3, 2, 3),
        (1, 2, 2, 3, 4, 5),
    ]

    # Configuration 3: use i in {0,2} for some blocks
    config3 = [
        (0, 2, 0, 1, 0, 1),
        (0, 2, 0, 1, 2, 3),
        (0, 2, 0, 1, 4, 5),
        (0, 2, 2, 3, 0, 1),
        (0, 2, 2, 3, 2, 3),
        (0, 2, 2, 3, 4, 5),
    ]

    configs = [config1, config2, config3]

    for cfg in configs:
        if time.time() > deadline:
            break
        rank, uc, vc, wc = try_configuration(cfg)
        if rank < best_rank:
            best_rank = rank
            best_U_cols = uc
            best_V_cols = vc
            best_W_cols = wc

    U_cols = best_U_cols
    V_cols = best_V_cols
    W_cols = best_W_cols
    rank = best_rank

    rank = len(U_cols)
    
    # Convert lists to proper matrix format
    U = [list(col) for col in zip(*U_cols)] if U_cols else []
    V = [list(col) for col in zip(*V_cols)] if V_cols else []
    W = [list(col) for col in zip(*W_cols)] if W_cols else []
    
    # Ensure correct shapes
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
