import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit

    n, m, p = DIMENSIONS
    rank = 88  # 32 (schoolbook 1x4x8) + 56 (Strassen 2x4x8)

    u_rows = n * m  # 12
    v_rows = m * p  # 32
    w_rows = p * n  # 24

    U = [[0] * rank for _ in range(u_rows)]
    V = [[0] * rank for _ in range(v_rows)]
    W = [[0] * rank for _ in range(w_rows)]

    col = 0

    # Part 1: i=0, schoolbook 1x4x8 = 32 terms
    for j in range(m):
        for k in range(p):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # Part 2: i=1,2, Strassen 2x4x8 = 56 terms
    # Strassen factors for 2x2x2 matrix multiplication
    # Local indices: A: (li,lj) -> li*2+lj, B: (lj,lk) -> lj*2+lk, C: (li,lk) -> li*2+lk
    strassen_U = [
        [1, 0, 0, 1],    # M1: A00+A11
        [0, 0, 1, 1],    # M2: A10+A11
        [1, 0, 0, 0],    # M3: A00
        [0, 0, 0, 1],    # M4: A11
        [1, 1, 0, 0],    # M5: A00+A01
        [-1, 0, 1, 0],   # M6: A10-A00
        [0, 1, 0, -1],   # M7: A11-A01
    ]
    strassen_V = [
        [1, 0, 0, 1],    # M1: B00+B11
        [1, 0, 0, 0],    # M2: B00
        [0, 1, 0, -1],   # M3: B10-B00
        [-1, 0, 1, 0],   # M4: B11-B01
        [0, 0, 0, 1],    # M5: B11
        [1, 1, 0, 0],    # M6: B00+B01
        [0, 0, 1, 1],    # M7: B10+B11
    ]
    strassen_W = [
        [1, 0, 0, 1],    # M1: C00+=M1, C11+=M1
        [0, 0, 1, -1],   # M2: C10+=M2, C11-=M2
        [0, 1, 0, 1],    # M3: C01+=M3, C11+=M3
        [1, 0, 1, 0],    # M4: C00+=M4, C10+=M4
        [-1, 1, 0, 0],   # M5: C00-=M5, C01+=M5
        [0, 0, 0, 1],    # M6: C11+=M6
        [1, 0, 0, 0],    # M7: C00+=M7
    ]

    for j_block in range(m // 2):  # 2 blocks
        for k_block in range(p // 2):  # 4 blocks
            for t in range(7):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

                # U factors: local (li, lj) -> global row (1+li, j_block*2+lj)
                for li in range(2):
                    for lj in range(2):
                        loc = li * 2 + lj
                        val = strassen_U[t][loc]
                        if val:
                            gi = 1 + li
                            gj = j_block * 2 + lj
                            U[gi * m + gj][col] += val

                # V factors: local (lj, lk) -> global row (j_block*2+lj, k_block*2+lk)
                for lj in range(2):
                    for lk in range(2):
                        loc = lj * 2 + lk
                        val = strassen_V[t][loc]
                        if val:
                            gj = j_block * 2 + lj
                            gk = k_block * 2 + lk
                            V[gj * p + gk][col] += val

                # W factors: local (li, lk) -> W index (gk*n + gi)
                for li in range(2):
                    for lk in range(2):
                        loc = li * 2 + lk
                        val = strassen_W[t][loc]
                        if val:
                            gi = 1 + li
                            gk = k_block * 2 + lk
                            W[gk * n + gi][col] += val

                col += 1

    return {"U": U, "V": V, "W": W}
