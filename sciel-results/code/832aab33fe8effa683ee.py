def construct():
    import time
    import numpy as np
    start = time.time()
    DEADLINE = 110  # safety margin before 120s (or 150s) kill

    n, m, p = 2, 4, 5
    # U: (n*m, rank) = (8, rank), row = i*m + j = i*4 + j
    # V: (m*p, rank) = (20, rank), row = j*p + k = j*5 + k
    # W: (p*n, rank) = (10, rank), row = k*n + i = k*2 + i

    U = {}  # row -> {col: val}
    V = {}
    W = {}
    next_col = 0

    # Strassen steps for 2x2x2
    # M1 = (A00+A11)(B00+B11) -> C00 + C11
    # M2 = (A10+A11)B00       -> C10 - C11
    # M3 = A00(B01-B11)       -> C01 + C11
    # M4 = A11(B10-B00)       -> C00 + C10
    # M5 = (A00+A01)B11       -> -C00 + C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00
    STRASSE = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    def add_24x42(a_cols, b_cols):
        """Add a 2x4 * 4x2 block using two 2x2 Strassen steps."""
        nonlocal next_col
        for part in range(2):
            for (X, Y, P) in STRASSE:
                col = next_col
                next_col += 1
                for (i, j), c in X.items():
                    r = i*4 + a_cols[part*2 + j]
                    U.setdefault(r, {})[col] = U.get(r, {}).get(col, 0) + c
                for (a, b), c in Y.items():
                    r = a_cols[part*2 + a]*5 + b_cols[b]
                    V.setdefault(r, {})[col] = V.get(r, {}).get(col, 0) + c
                for (i, b), c in P.items():
                    r = b_cols[b]*2 + i
                    W.setdefault(r, {})[col] = W.get(r, {}).get(col, 0) + c

    def add_24x41(a_cols, b_col):
        """Add a 2x4 * 4x1 block using schoolbook (8 rank-1 terms)."""
        nonlocal next_col
        for part in range(2):
            for l_local in range(2):
                for i in range(2):
                    col = next_col
                    next_col += 1
                    r_u = i*4 + a_cols[part*2 + l_local]
                    U.setdefault(r_u, {})[col] = U.get(r_u, {}).get(col, 0) + 1
                    r_v = a_cols[part*2 + l_local]*5 + b_col
                    V.setdefault(r_v, {})[col] = V.get(r_v, {}).get(col, 0) + 1
                    r_w = b_col*2 + i
                    W.setdefault(r_w, {})[col] = W.get(r_w, {}).get(col, 0) + 1

    # Sub-problems:
    # C2: A * B[:, 2:4]  (2x4 * 4x2) -> 14 rank terms (two Strassen blocks)
    add_24x42([0,1,2,3], [2,3])
    # C3: A * B[:, 4:5]  (2x4 * 4x1) -> 8 rank terms (schoolbook)
    add_24x41([0,1,2,3], 4)
    # C4: A * B[:, 0:2]  (2x4 * 4x2) -> 14 rank terms (two Strassen blocks)
    add_24x42([0,1,2,3], [0,1])

    rank = next_col

    # Convert to numpy arrays for efficient verification
    U_arr = np.zeros((8, rank), dtype=float)
    V_arr = np.zeros((20, rank), dtype=float)
    W_arr = np.zeros((10, rank), dtype=float)
    for r, cols in U.items():
        for c, val in cols.items():
            U_arr[r, c] = val
    for r, cols in V.items():
        for c, val in cols.items():
            V_arr[r, c] = val
    for r, cols in W.items():
        for c, val in cols.items():
            W_arr[r, c] = val

    def verify_decomposition(U_mat, V_mat, W_mat, r):
        """Check if the decomposition is exact for all (i,j,k) in [0,n)x[0,m)x[0,p)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = i * m + j
                    v_row = j * p + k
                    w_row = k * n + i
                    val = np.sum(U_mat[u_row, :r] * V_mat[v_row, :r] * W_mat[w_row, :r])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try to reduce rank by removing columns and checking validity
    best_rank = rank
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()

    # First, try removing single columns
    for col_to_remove in range(rank):
        if time.time() - start > DEADLINE - 10:
            break
        r = rank - 1
        # Create masked arrays
        mask = np.ones(rank, dtype=bool)
        mask[col_to_remove] = False
        U_test = U_arr[:, mask]
        V_test = V_arr[:, mask]
        W_test = W_arr[:, mask]
        
        if verify_decomposition(U_test, V_test, W_test, r):
            best_rank = r
            best_U = U_test.copy()
            best_V = V_test.copy()
            best_W = W_test.copy()

    # If we found a reduction, try further reductions on the reduced set
    if best_rank < rank:
        current_rank = best_rank
        U_cur = best_U
        V_cur = best_V
        W_cur = best_W
        
        while True:
            if time.time() - start > DEADLINE - 5:
                break
            found_removal = False
            for col_to_remove in range(current_rank):
                r = current_rank - 1
                mask = np.ones(current_rank, dtype=bool)
                mask[col_to_remove] = False
                U_test = U_cur[:, mask]
                V_test = V_cur[:, mask]
                W_test = W_cur[:, mask]
                
                if verify_decomposition(U_test, V_test, W_test, r):
                    best_rank = r
                    best_U = U_test.copy()
                    best_V = V_test.copy()
                    best_W = W_test.copy()
                    current_rank = r
                    U_cur = U_test
                    V_cur = V_test
                    W_cur = W_test
                    found_removal = True
                    break
            
            if not found_removal:
                break

    # Convert back to lists
    final_rank = best_rank
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    # Ensure integer/half-integer entries are clean
    for r in range(len(U_list)):
        for c in range(final_rank):
            if abs(U_list[r][c] - round(U_list[r][c])) < 1e-9:
                U_list[r][c] = round(U_list[r][c])
    for r in range(len(V_list)):
        for c in range(final_rank):
            if abs(V_list[r][c] - round(V_list[r][c])) < 1e-9:
                V_list[r][c] = round(V_list[r][c])
    for r in range(len(W_list)):
        for c in range(final_rank):
            if abs(W_list[r][c] - round(W_list[r][c])) < 1e-9:
                W_list[r][c] = round(W_list[r][c])

    return {"U": U_list, "V": V_list, "W": W_list}
