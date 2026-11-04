"""Strassen-based rank-44 decomposition for 2x3x8 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # Schoolbook fallback (rank 48)
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
                u_row = U[i * m + a]
                for k in range(p):
                    acc = 0
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        u_val = u_row[c]
                        if u_val:
                            v_val = v_row[c]
                            if v_val:
                                acc += u_val * v_val * w_row[c]
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition vectors
    # U vectors: for A[i,a] where i in {0,1}, a in {0,1}
    S_U = [
        [1, 0, 0, 1],   # m1: (a00+a11)
        [0, 0, 1, 1],   # m2: (a10+a11)
        [1, 0, 0, 0],   # m3: a00
        [0, 0, 0, 1],   # m4: a11
        [1, 1, 0, 0],   # m5: (a00+a10)
        [-1, 0, 1, 0],  # m6: (a11-a10)
        [0, 1, 0, -1]   # m7: (a01-a11)
    ]
    # V vectors: for B[a,k] where a in {0,1}, k in {0,1}
    S_V = [
        [1, 0, 0, 1],   # m1: (b00+b11)
        [1, 0, 0, 0],   # m2: b00
        [0, 1, 0, -1],  # m3: (b10-b11)
        [-1, 0, 1, 0],  # m4: (b01-b00)
        [0, 0, 0, 1],   # m5: b11
        [1, 1, 0, 0],   # m6: (b00+b10)
        [0, 0, 1, 1]    # m7: (b01+b11)
    ]
    # W vectors: for C[k,i] where k in {0,1}, i in {0,1}
    # C00 = m1 + m4 - m5 - m7
    # C01 = m2 + m7
    # C10 = m3 + m5
    # C11 = m1 - m2 + m3 + m6
    S_W = [
        [1, 0, 0, 1],   # m1: C00 += 1, C11 += 1
        [0, 1, 0, -1],  # m2: C01 += 1, C11 -= 1
        [0, 0, 1, 1],   # m3: C10 += 1, C11 += 1
        [1, 1, 0, 0],   # m4: C00 += 1, C01 += 1
        [-1, 0, 1, 0],  # m5: C00 -= 1, C10 += 1
        [0, 0, 0, 1],   # m6: C11 += 1
        [1, 0, 0, 0]    # m7: C00 += 1, C01 += 1
    ]

    # Verify Strassen correctness
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

    if not _verify_strassen():
        return _schoolbook()

    # Build rank-44 candidate
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8) using Strassen on 4 column blocks of 2
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            # Fill U for this Strassen term
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

            # Fill V for this Strassen term
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

            # Fill W for this Strassen term
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8) schoolbook
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
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

# EVOLVE-BLOCK-END
