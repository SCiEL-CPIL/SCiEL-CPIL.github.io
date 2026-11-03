import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin under 150s limit

    n, m, p = DIMENSIONS
    # U: (n*m, rank) -> (16, rank)
    # V: (m*p, rank) -> (24, rank)
    # W: (p*n, rank) -> (24, rank)  with W[k*n+i, r]

    def time_left():
        return (deadline - time.time()) > 1.0

    # --- Build the rank-77 base decomposition ---
    rank = 77
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for i, c in u_coeffs:
            if c != 0:
                U[i, col] += c
        for j, c in v_coeffs:
            if c != 0:
                V[j, col] += c
        for k, c in w_coeffs:
            if c != 0:
                W[k, col] += c
        col += 1

    # Index helpers
    def a_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x * 2 + r) * 4 + (y * 2 + c)

    def b_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x * 2 + r) * 6 + (y * 2 + c)

    def c_idx(block, r, c):
        # Output index: W[k*n+i] where k is output col (1..p), i is output row (1..n)
        x, y = int(block[0]), int(block[1])  # x = row block, y = col block
        return (y * 2 + c) * 4 + (x * 2 + r)

    # Strassen 2x2x2 scalar terms (7): each is (u_coeffs, v_coeffs)
    def get_strassen_2x2_terms():
        s1 = ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)])
        s2 = ([((1,0), 1), ((1,1), 1)], [((0,0), 1)])
        s3 = ([((0,0), 1)], [((0,1), 1), ((1,1), -1)])
        s4 = ([((1,1), 1)], [((1,0), 1), ((0,0), -1)])
        s5 = ([((0,0), 1), ((0,1), 1)], [((1,1), 1)])
        s6 = ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)])
        s7 = ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)])
        return [s1, s2, s3, s4, s5, s6, s7]

    strassen_2x2_terms = get_strassen_2x2_terms()

    # Result mapping for 2x2 Strassen: output block pos -> list of (term_idx, coeff)
    result_map = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }

    # Big Strassen 4x4 combinations (7 big multiplies)
    big_combos = [
        ([('00', 1), ('11', 1)], [('00', 1), ('11', 1)]), # M1
        ([('10', 1), ('11', 1)], [('00', 1)]),            # M2
        ([('00', 1)], [('01', 1), ('11', -1)]),           # M3
        ([('11', 1)], [('10', 1), ('00', -1)]),           # M4
        ([('00', 1), ('01', 1)], [('11', 1)]),            # M5
        ([('10', 1), ('00', -1)], [('00', 1), ('01', 1)]),# M6
        ([('01', 1), ('11', -1)], [('10', 1), ('11', 1)]) # M7
    ]

    big_result_map = {
        '00': [(0, 1), (3, 1), (4, -1), (6, 1)],
        '01': [(2, 1), (4, 1)],
        '10': [(1, 1), (3, 1)],
        '11': [(0, 1), (1, -1), (2, 1), (5, 1)]
    }

    # Part 1: Strassen for 4x4x4 (B cols 0-3, C cols 0-3) -> 49 terms
    for big_idx, (a_parts, b_parts) in enumerate(big_combos):
        if not time_left():
            break
        for s_idx in range(7):
            if not time_left():
                break
            x_coeffs, y_coeffs = strassen_2x2_terms[s_idx]

            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                for a_block, a_sign in a_parts:
                    idx = a_idx(a_block, rx, cx)
                    u_coeffs.append((idx, a_sign * cx_c))

            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                for b_block, b_sign in b_parts:
                    idx = b_idx(b_block, ry, cy)
                    v_coeffs.append((idx, b_sign * cy_c))

            w_coeffs = []
            for c_block, big_contribs in big_result_map.items():
                big_coeff = 0
                for m_idx, m_c in big_contribs:
                    if m_idx == big_idx:
                        big_coeff = m_c
                        break
                if big_coeff == 0:
                    continue
                for rc, s_contribs in result_map.items():
                    small_coeff = 0
                    for s_check, s_c in s_contribs:
                        if s_check == s_idx:
                            small_coeff = s_c
                            break
                    if small_coeff == 0:
                        continue
                    idx = c_idx(c_block, rc[0], rc[1])
                    w_coeffs.append((idx, big_coeff * small_coeff))

            add_term(u_coeffs, v_coeffs, w_coeffs)

    # Part 2: Schoolbook 4x4x2 (B cols 4-5, C cols 4-5) -> 32 terms
    products = [
        ('00', 0, 0),  # A top-left block, B rows 0-1, C cols 0-1
        ('01', 0, 2),  # A top-right block, B rows 0-1, C cols 2-3
        ('10', 2, 0),  # A bottom-left block, B rows 2-3, C cols 0-1
        ('11', 2, 2)   # A bottom-right block, B rows 2-3, C cols 2-3
    ]

    for a_block, b_row_base, c_col_base in products:
        if not time_left():
            break
        x, y = int(a_block[0]), int(a_block[1])
        for ri in range(2):
            for bi in range(2):
                u_idx = (x * 2 + ri) * 4 + (y * 2 + bi)
                v_idx = (b_row_base + bi) * 6 + 4
                w_idx = (4 + c_col_base) * 4 + (x * 2 + ri)
                add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    # Trim to actually used columns
    if col < rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]

    # --- Redundancy removal: find and drop columns that are linear combos ---
    if time_left():
        # Column space of U, V, W
        rank_U = np.linalg.matrix_rank(U, tol=1e-9)
        rank_V = np.linalg.matrix_rank(V, tol=1e-9)
        rank_W = np.linalg.matrix_rank(W, tol=1e-9)
        min_rank = min(rank_U, rank_V, rank_W)

        # Try to identify redundant columns by checking if removing column j
        # still keeps all three column spaces at >= min_rank
        # Start from the full set and greedily remove columns
        keep = list(range(col))
        # Sort columns by "importance" (norm) to try removing low-norm first
        col_norms = np.linalg.norm(U, axis=0) + np.linalg.norm(V, axis=0) + np.linalg.norm(W, axis=0)
        order = np.argsort(col_norms)

        for j in order:
            if not time_left():
                break
            j = int(j)
            if j not in keep:
                continue
            new_keep = [k for k in keep if k != j]
            if len(new_keep) < min_rank:
                continue
            Uk = U[:, new_keep]
            Vk = V[:, new_keep]
            Wk = W[:, new_keep]
            if np.linalg.matrix_rank(Uk, tol=1e-9) < min_rank:
                continue
            if np.linalg.matrix_rank(Vk, tol=1e-9) < min_rank:
                continue
            if np.linalg.matrix_rank(Wk, tol=1e-9) < min_rank:
                continue
            keep = new_keep

        if len(keep) < col:
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]

    U = U.tolist()
    V = V.tolist()
    W = W.tolist()

    return {"U": U, "V": V, "W": W}
