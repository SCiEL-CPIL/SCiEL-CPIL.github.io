import time

def construct():
    t0 = time.time()
    deadline = t0 + 135.0
    
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
    # U rows: i*m+a (i in 0..1, a in 0..1)
    # V rows: a*p+k (a in 0..1, k in 0..1)
    # W rows: k*n+i (k in 0..1, i in 0..1)
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8), split B1 into 4 blocks of 2
    # A1 corresponds to rows 0..3 of U (i in 0..1, a in 0..1)
    # B1 corresponds to rows 0..15 of V (a in 0..1, k in 0..7)
    # C1 corresponds to rows 0..15 of W (k in 0..7, i in 0..1)
    
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()
    
    # Part 2: A2 (2x1) * B2 (1x8)
    # A2 corresponds to rows 4..5 of U (i in 0..1, a=2)
    # B2 corresponds to rows 16..23 of V (a=2, k in 0..7)
    # C2 corresponds to rows 0..15 of W (k in 0..7, i in 0..1) -- Wait, W indices are k*n+i.
    # The output C is 2x8. C[i,k].
    # For Part 1, we filled C[i, k] for k in 0..7.
    # For Part 2, we also fill C[i, k] for k in 0..7.
    # So W rows are the same indices.
    
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()
    
    # Validation
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += u_row[c] * v_row[c] * w_row[c]
                    if acc != 1:
                        return False
        return True
    
    candidate = {"U": U, "V": V, "W": W}
    if _validate(candidate):
        return candidate
    return _schoolbook()
