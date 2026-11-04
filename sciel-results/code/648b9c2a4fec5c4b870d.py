import time

def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

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

    # Strassen 2x2x2
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

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8), split B1 into 4 blocks of 2 columns
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            # Unroll loops for speed and clarity
            # U indices: li in 0..1, la in 0..1 -> li*3 + la
            # V indices: la in 0..1, lk in 0..1 -> la*8 + (k0+lk)
            # W indices: lk in 0..1, li in 0..1 -> (k0+lk)*2 + li
            
            u_vals = S_U[r]
            v_vals = S_V[r]
            w_vals = S_W[r]
            
            # li=0, la=0
            if u_vals[0]: U[0][col] += u_vals[0]
            # li=0, la=1
            if u_vals[1]: U[1][col] += u_vals[1]
            # li=1, la=0
            if u_vals[2]: U[3][col] += u_vals[2]
            # li=1, la=1
            if u_vals[3]: U[4][col] += u_vals[3]

            # la=0, lk=0
            if v_vals[0]: V[k0][col] += v_vals[0]
            # la=0, lk=1
            if v_vals[1]: V[k0+1][col] += v_vals[1]
            # la=1, lk=0
            if v_vals[2]: V[8+k0][col] += v_vals[2]
            # la=1, lk=1
            if v_vals[3]: V[8+k0+1][col] += v_vals[3]

            # lk=0, li=0
            if w_vals[0]: W[k0*2][col] += w_vals[0]
            # lk=0, li=1
            if w_vals[1]: W[k0*2+1][col] += w_vals[1]
            # lk=1, li=0
            if w_vals[2]: W[(k0+1)*2][col] += w_vals[2]
            # lk=1, li=1
            if w_vals[3]: W[(k0+1)*2+1][col] += w_vals[3]

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8)
    # U indices: i in 0..1, a=2 -> i*3 + 2 => 2, 5
    # V indices: a=2, k in 0..7 -> 2*8 + k => 16..23
    # W indices: k in 0..7, i in 0..1 -> k*2 + i
    for i in range(2):
        u_idx = i * 3 + 2
        for k in range(8):
            v_idx = 16 + k
            w_idx = k * 2 + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()
