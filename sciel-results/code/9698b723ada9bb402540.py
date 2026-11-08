import time

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    
    # Schoolbook fallback (Rank 48)
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
    # U factors (4x7), V factors (4x7), W factors (4x7)
    # Indices: U[i*2+j], V[i*2+j], W[i*2+j]
    # A is 2x2, B is 2x2, C is 2x2
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8)
    # B1 is rows 0,1 of B (indices 0..7 in V)
    # A1 is cols 0,1 of A (indices 0..3 in U)
    # C1 is cols 0,1 of C? No, C is 2x8.
    # C = A1 B1 + A2 B2
    # B1 is 2x8. We split B1 into 4 blocks of 2x2: B10, B11, B12, B13
    # B10: B[0,0:2], B[1,0:2]
    # B11: B[0,2:4], B[1,2:4]
    # etc.
    
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            # U part: A1 (2x2) -> indices 0,1,2,3
            # U[li * m + la] where li in 0..1, la in 0..1
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val
            
            # V part: B1 block (2x2) -> rows 0,1 of B, cols k0, k0+1
            # V[a * p + k] where a in 0..1, k in k0..k0+1
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val
            
            # W part: C block (2x2) -> rows k0, k0+1 of C, cols 0,1 of C (which are 0,1)
            # W[k * n + i] where k in k0..k0+1, i in 0..1
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val
            
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8)
    # A2 is col 2 of A -> indices 4,5 in U
    # B2 is row 2 of B -> indices 16..23 in V
    # C2 is all of C (2x8)
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    return {"U": U, "V": V, "W": W}
