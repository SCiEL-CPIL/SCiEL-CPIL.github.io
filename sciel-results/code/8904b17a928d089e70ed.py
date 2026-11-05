import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 4, 4, 7
    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_2x2_block(i_off, j_off, k_off):
        a = 1
        # m1 = (a11+a22)*(b11+b22)
        u = [0] * (n * m)
        u[i_off * m + j_off] = a
        u[(i_off+1) * m + (j_off+1)] = a
        v = [0] * (m * p)
        v[j_off * p + k_off] = a
        v[(j_off+1) * p + (k_off+1)] = a
        w = [0] * (p * n)
        w[k_off * n + i_off] = a
        w[(k_off+1) * n + (i_off+1)] = a
        add_term(u, v, w)

        # m2 = (a21+a22)*b11
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a
        u[(i_off+1) * m + (j_off+1)] = a
        v = [0] * (m * p)
        v[j_off * p + k_off] = a
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = a
        w[(k_off+1) * n + (i_off+1)] = -a
        add_term(u, v, w)

        # m3 = a11*(b12-b22)
        u = [0] * (n * m)
        u[i_off * m + j_off] = a
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = a
        v[(j_off+1) * p + (k_off+1)] = -a
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = a
        w[(k_off+1) * n + (i_off+1)] = a
        add_term(u, v, w)

        # m4 = a22*(b21-b11)
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = a
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = a
        v[j_off * p + k_off] = -a
        w = [0] * (p * n)
        w[k_off * n + i_off] = a
        w[k_off * n + (i_off+1)] = a
        add_term(u, v, w)

        # m5 = (a11+a12)*b22
        u = [0] * (n * m)
        u[i_off * m + j_off] = a
        u[i_off * m + (j_off+1)] = a
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = a
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = a
        w[k_off * n + i_off] = -a
        add_term(u, v, w)

        # m6 = (a21-a11)*(b11+b12)
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a
        u[i_off * m + j_off] = -a
        v = [0] * (m * p)
        v[j_off * p + k_off] = a
        v[j_off * p + (k_off+1)] = a
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = a
        add_term(u, v, w)

        # m7 = (a12-a22)*(b21+b22)
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = a
        u[(i_off+1) * m + (j_off+1)] = -a
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = a
        v[(j_off+1) * p + (k_off+1)] = a
        w = [0] * (p * n)
        w[k_off * n + i_off] = a
        add_term(u, v, w)

    def strassen_4x4_head():
        # 4x4x4 Strassen decomposition using 7 terms
        # Each term is a Kronecker product of 4x4 block structure and 2x2 Strassen
        outer_terms = [
            # (A_pattern, B_pattern, C_pattern)
            # Each pattern is a list of (block_idx, coeff) where block_idx encodes position
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # m1
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),         # m2
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),         # m3
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),         # m4
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),         # m5
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),         # m6
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),         # m7
        ]

        inner_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        def get_A_idx(block_idx, local_idx):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc
            return row * m + col

        def get_B_idx(block_idx, local_idx):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc
            return row * p + col

        def get_C_idx(block_idx, local_idx):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc
            return col * n + row

        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_A_idx(a_block, x_local)
                        u[idx] += a_coeff * x_coeff
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_B_idx(b_block, y_local)
                        v[idx] += b_coeff * y_coeff
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_C_idx(c_block, z_local)
                        w[idx] += c_coeff * z_coeff
                add_term(u, v, w)

    def strassen_4x4x2_tail():
        # For columns 4,5 of B (k=4,5), use 2x2 Strassen for each 2x2 block of A
        for i_b in range(2):
            for j_b in range(2):
                i_off = i_b * 2
                j_off = j_b * 2
                k_off = 4
                strassen_2x2_block(i_off, j_off, k_off)

    def schoolbook_4x4x1_tail():
        # For column 6 of B (k=6), use schoolbook
        for i in range(n):
            for j in range(m):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                add_term(u, v, w)

    # Build the decomposition
    strassen_4x4_head()
    strassen_4x4x2_tail()
    schoolbook_4x4x1_tail()

    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
