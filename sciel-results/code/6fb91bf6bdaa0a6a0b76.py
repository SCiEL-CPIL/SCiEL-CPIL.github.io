import time

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    n, m, p = 2, 3, 8

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Strassen 2x2x2 decomposition
    # U index: i*2+a (i in 0,1; a in 0,1)
    # V index: a*2+k (a in 0,1; k in 0,1)
    # W index: k*2+i (k in 0,1; i in 0,1)
    S_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    S_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    S_W = [
        [1, 0, 0, 1],   # M1
        [0, 1, 0, -1],  # M2
        [0, 0, 1, 1],   # M3
        [1, 1, 0, 0],   # M4
        [-1, 0, 1, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]

    # Strassen-like 2x1x2 decomposition (rank 3)
    # A is 2x1 (rows i=0,1), B is 1x2 (cols k=0,1), C is 2x2 (k*n+i)
    # U index: i (0,1)
    # V index: k (0,1)
    # W index: k*2+i
    S212_U = [
        [1, 0, 1],   # M1: A[0]
        [0, 1, 1],   # M2: A[1]
        [0, 0, 1]    # M3: A[0]+A[1]
    ]
    S212_V = [
        [1, 0, 1],   # M1: B[0]
        [0, 1, 1],   # M2: B[1]
        [1, 1, 0]    # M3: B[0]+B[1]
    ]
    S212_W = [
        [1, 0, 0, 0],   # M1: C[0,0]
        [0, 0, 1, 0],   # M2: C[1,1]
        [0, 1, 0, 1]    # M3: C[0,1] + C[1,0]
    ]

    # Verify 2x1x2 decomposition
    def _verify_212():
        for i in range(2):
            for k in range(2):
                acc = 0
                for r in range(3):
                    u_val = S212_U[r][i]
                    v_val = S212_V[r][k]
                    w_val = S212_W[r][k * 2 + i]
                    acc += u_val * v_val * w_val
                if acc != 1:
                    return False
        return True

    # Verify Strassen 2x2x2
    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen() or not _verify_212():
        return _schoolbook()

    # Total rank: 4 blocks * 7 (Strassen 2x2x2) + 4 blocks * 3 (Strassen 2x1x2) = 40
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8), split B1 cols into 4 blocks of 2
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            # U: local (li, la) -> global row li*m + la
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val
            # V: local (la, lk) -> global row la*p + (k0+lk)
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val
            # W: local (lk, li) -> global row (k0+lk)*n + li
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8), split B2 cols into 4 blocks of 2
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(3):
            # U: row i (global i*m + 2)
            for li in range(2):
                u_val = S212_U[r][li]
                if u_val != 0:
                    U[li * m + 2][col] = u_val
            # V: row 2*p + (k0+lk)
            for lk in range(2):
                v_val = S212_V[r][lk]
                if v_val != 0:
                    V[2 * p + (k0 + lk)][col] = v_val
            # W: row (k0+lk)*n + li
            for lk in range(2):
                for li in range(2):
                    w_val = S212_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Validate using integer arithmetic
    def _validate(candidate):
        U_mat = candidate["U"]
        V_mat = candidate["V"]
        W_mat = candidate["W"]
        rank_val = len(U_mat[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U_mat[i * m + a]
                    v_row = V_mat[a * p + k]
                    w_row = W_mat[k * n + i]
                    for c in range(rank_val):
                        uc = u_row[c]
                        vc = v_row[c]
                        wc = w_row[c]
                        if uc != 0 and vc != 0 and wc != 0:
                            acc += uc * vc * wc
                    if acc != 1:
                        return False
        return True

    candidate = {"U": U, "V": V, "W": W}
    if _validate(candidate):
        return candidate
    return _schoolbook()
