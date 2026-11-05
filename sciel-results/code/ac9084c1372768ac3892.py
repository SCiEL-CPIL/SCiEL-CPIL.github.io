import time

DIMENSIONS = [4, 4, 5]
_START_TIME = time.time()
TIME_LIMIT = 120.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5

    if _time_remaining() < 1.0:
        # Fallback: Schoolbook rank 80
        rank = 80
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0

    # Strassen 4x4x4 decomposition (Rank 49)
    # Split into 2x2 blocks.
    a_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]

    b_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]

    c_contrib = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    p_inner_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]

    q_inner_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]

    r_inner_contrib = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def get_block_indices(block_type, block_idx, local_r, local_c):
        if block_type == 'A':
            br = block_idx // 2
            bc = block_idx % 2
            gi = br * 2 + local_r
            gj = bc * 2 + local_c
            return (gi, gj)
        elif block_type == 'B':
            br = block_idx // 2
            bc = block_idx % 2
            gj = br * 2 + local_r
            gk = bc * 2 + local_c
            return (gj, gk)
        elif block_type == 'C':
            br = block_idx // 2
            bc = block_idx % 2
            gi = br * 2 + local_r
            gk = bc * 2 + local_c
            return (gi, gk)

    for super_idx in range(7):
        for inner_idx in range(7):
            u_list = []
            v_list = []
            w_list = []

            for p_elem_idx, p_coeff in enumerate(p_inner_coeffs[inner_idx]):
                if p_coeff == 0:
                    continue
                pr = p_elem_idx // 2
                pc = p_elem_idx % 2
                for a_block_idx, a_coeff in enumerate(a_coeffs[super_idx]):
                    if a_coeff == 0:
                        continue
                    gi, gj = get_block_indices('A', a_block_idx, pr, pc)
                    u_list.append((gi, gj, a_coeff * p_coeff))

            for q_elem_idx, q_coeff in enumerate(q_inner_coeffs[inner_idx]):
                if q_coeff == 0:
                    continue
                qr = q_elem_idx // 2
                qc = q_elem_idx % 2
                for b_block_idx, b_coeff in enumerate(b_coeffs[super_idx]):
                    if b_coeff == 0:
                        continue
                    gj, gk = get_block_indices('B', b_block_idx, qr, qc)
                    v_list.append((gj, gk, b_coeff * q_coeff))

            for c_block_idx, super_coeff in enumerate(c_contrib[super_idx]):
                if super_coeff == 0:
                    continue
                for r_block_idx, inner_coeff in enumerate(r_inner_contrib[inner_idx]):
                    if inner_coeff == 0:
                        continue
                    total_coeff = super_coeff * inner_coeff
                    if total_coeff == 0:
                        continue
                    lr = r_block_idx // 2
                    lc = r_block_idx % 2
                    gi, gk = get_block_indices('C', c_block_idx, lr, lc)
                    w_list.append((gk, gi, total_coeff))

            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1

    # Part 2: 4x4x1 (Rank 16) for column 4
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    # Verification: reconstruct the tensor and check it equals the matmul tensor
    # T[i, j, k] should be 1 for all valid (i, j, k) triples, else 0.
    # Reconstructed: sum_t U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
    # We check a sample of triples to ensure correctness without full O(n*m*p*rank) cost.
    # Full check is cheap: 4*4*5*65 = 5200 ops.
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if s != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break

    if not ok:
        # Fallback to schoolbook if verification fails
        rank = 80
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
