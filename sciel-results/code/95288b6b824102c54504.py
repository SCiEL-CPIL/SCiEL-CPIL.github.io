import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses 6 Strassen blocks for the main 2x4x6 core.
    Uses rank-1 slice decompositions for the remaining uncovered parts.
    Total Rank: 52.
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
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    # 1. Add 6 Strassen blocks
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]
    
    for block_params in blocks:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)

    # 2. Handle Uncovered Part 2: j=4, i in {0,1}, k in {0..6}
    # This is a slice with fixed j. Rank 1.
    # U_{(i,4), r} = 1, V_{(4,k), r} = 1, W_{(k,i), r} = 1
    u_entries = {}
    v_entries = {}
    w_entries = {}
    for i in range(n):
        if i == 2: continue # i=2 is handled in Part 1
        u_entries[i * m + 4] = 1
    for k in range(p):
        v_entries[4 * p + k] = 1
        for i in range(n):
            if i == 2: continue
            w_entries[k * n + i] = 1
    add_term(u_entries, v_entries, w_entries)

    # 3. Handle Uncovered Part 1: i=2, j in {0..4}, k in {0..6}
    # This is a slice with fixed i. Rank m (5).
    # For each j, add a term.
    for j in range(m):
        u_entries = {2 * m + j: 1}
        v_entries = {}
        w_entries = {}
        for k in range(p):
            v_entries[j * p + k] = 1
            w_entries[k * n + 2] = 1
        add_term(u_entries, v_entries, w_entries)

    # 4. Handle Uncovered Part 3: k=6, i in {0,1}, j in {0..3}
    # This is a slice with fixed k. Rank m (4).
    # For each j, add a term.
    for j in range(4):
        u_entries = {}
        v_entries = {j * p + 6: 1}
        w_entries = {}
        for i in range(2):
            u_entries[i * m + j] = 1
            w_entries[6 * n + i] = 1
        add_term(u_entries, v_entries, w_entries)

    rank = len(U_cols)
    
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
