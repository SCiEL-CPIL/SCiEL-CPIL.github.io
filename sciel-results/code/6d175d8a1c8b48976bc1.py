import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 135

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Strassen 2x2*2x2 decomposition (rank 7)
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))

    # c=0: M1 = (a00+a11)(b00+b11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1

    # c=1: M2 = (a10+a11)*b00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1

    # c=2: M3 = a00*(b01-b11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1

    # c=3: M4 = a11*(b10-b00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1

    # c=4: M5 = (a00+a01)*b11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1

    # c=5: M6 = (a10-a00)*(b00+b01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1

    # c=6: M7 = (a01-a11)*(b10+b11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Block Strassen for 4x4*4x4 (rank 49)
    A_specs = [
        [(1, 0, 0), (1, 1, 1)],
        [(1, 1, 0), (1, 1, 1)],
        [(1, 0, 0)],
        [(1, 1, 1)],
        [(1, 0, 0), (1, 0, 1)],
        [(1, 1, 0), (-1, 0, 0)],
        [(1, 0, 1), (-1, 1, 1)],
    ]

    B_specs = [
        [(1, 0, 0), (1, 1, 1)],
        [(1, 0, 0)],
        [(1, 0, 1), (-1, 1, 1)],
        [(1, 1, 0), (-1, 0, 0)],
        [(1, 1, 1)],
        [(1, 0, 0), (1, 0, 1)],
        [(1, 1, 0), (1, 1, 1)],
    ]

    C_specs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_dict, v_dict, w_dict):
        U_row = np.zeros(nm)
        V_row = np.zeros(mp)
        W_row = np.zeros(pn)
        for (i, j), val in u_dict.items():
            U_row[i * m + j] += val
        for (j, k), val in v_dict.items():
            V_row[j * p + k] += val
        for (k, i), val in w_dict.items():
            W_row[k * n + i] += val
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)

    def strassen_4x4x4(A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off):
        for k_p in range(7):
            for m_s in range(7):
                u_dict = {}
                v_dict = {}
                w_dict = {}

                for (coeff, br, bc) in A_specs[k_p]:
                    for r in range(2):
                        for c in range(2):
                            u_val = U_2[r * 2 + c, m_s]
                            if u_val != 0:
                                gi = A_row_off + br * 2 + r
                                gj = A_col_off + bc * 2 + c
                                u_dict[(gi, gj)] = u_dict.get((gi, gj), 0) + coeff * u_val

                for (coeff, br, bc) in B_specs[k_p]:
                    for r in range(2):
                        for c in range(2):
                            v_val = V_2[r * 2 + c, m_s]
                            if v_val != 0:
                                gj = B_row_off + br * 2 + r
                                gk = B_col_off + bc * 2 + c
                                v_dict[(gj, gk)] = v_dict.get((gj, gk), 0) + coeff * v_val

                for (cbr, cbc), coeff_c in C_specs[k_p].items():
                    for r in range(2):
                        for c in range(2):
                            w_val = W_2[r * 2 + c, m_s]
                            if w_val != 0:
                                gi = C_row_off + cbr * 2 + r
                                gk = C_row_off + cbc * 2 + c
                                w_dict[(gk, gi)] = w_dict.get((gk, gi), 0) + coeff_c * w_val

                if u_dict and v_dict and w_dict:
                    add_term(u_dict, v_dict, w_dict)

    # Apply for B_left (cols 0-3) and B_right (cols 4-7)
    strassen_4x4x4(0, 0, 0, 0, 0, 0)
    strassen_4x4x4(0, 0, 0, 4, 0, 4)

    rank = len(U_list)
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    # Precompute indices for fast verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_flat_idx = i_idx * m + j_idx
    v_flat_idx = j_idx * p + k_idx
    w_flat_idx = k_idx * n + i_idx

    def verify_fast(U_mat, V_mat, W_mat):
        prod = U_mat[u_flat_idx, :] * V_mat[v_flat_idx, :] * W_mat[w_flat_idx, :]
        tensor_vals = prod.sum(axis=1)
        return np.max(np.abs(tensor_vals - 1.0))

    error = verify_fast(U, V, W)
    if error > 1e-8:
        # Fallback to schoolbook
        U_school = np.zeros((nm, n * m * p))
        V_school = np.zeros((mp, n * m * p))
        W_school = np.zeros((pn, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col] = 1
                    V_school[j*p + k, col] = 1
                    W_school[k*n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Aggressive rank reduction
    current_U, current_V, current_W = U, V, W
    current_rank = rank
    removed = set()

    def try_remove_term(U_mat, V_mat, W_mat, r, removed_set):
        """Try removing term r; return (success, new_rank)."""
        keep_mask = np.ones(len(U_mat), dtype=bool)
        keep_mask[r] = False
        for rm in removed_set:
            if rm < len(U_mat):
                keep_mask[rm] = False
        n_keep = keep_mask.sum()
        if n_keep == 0:
            return False, 0
        U_test = U_mat[:, keep_mask]
        V_test = V_mat[:, keep_mask]
        W_test = W_mat[:, keep_mask]
        err = verify_fast(U_test, V_test, W_test)
        return err < 1e-7, n_keep

    # Pass 1: Remove terms in increasing norm order (most likely redundant first)
    if time.time() < deadline - 20:
        norms = np.array([
            np.linalg.norm(current_U[:, r]) * np.linalg.norm(current_V[:, r]) * np.linalg.norm(current_W[:, r])
            for r in range(current_rank)
        ])
        order = np.argsort(norms)
        for r in order:
            if r in removed or time.time() > deadline - 20:
                continue
            ok, _ = try_remove_term(current_U, current_V, current_W, r, removed)
            if ok:
                removed.add(r)
                current_rank -= 1

    # Pass 2: Randomized removal attempts
    if time.time() < deadline - 20:
        rng = np.random.default_rng(42)
        for _ in range(20):
            if time.time() > deadline - 20:
                break
            remaining = [r for r in range(rank) if r not in removed]
            if not remaining:
                break
            r = rng.choice(remaining)
            ok, _ = try_remove_term(current_U, current_V, current_W, r, removed)
            if ok:
                removed.add(r)
                current_rank -= 1

    # Pass 3: Try pairs of removals (sometimes removing two terms works when neither alone does)
    if time.time() < deadline - 15 and len(removed) < current_rank - 10:
        remaining = [r for r in range(rank) if r not in removed]
        if len(remaining) >= 2:
            import itertools
            pairs = list(itertools.combinations(remaining, 2))
            rng = np.random.default_rng(123)
            rng.shuffle(pairs)
            for r1, r2 in pairs:
                if time.time() > deadline - 15:
                    break
                if r1 in removed or r2 in removed:
                    continue
                # Try removing both
                keep_mask = np.ones(rank, dtype=bool)
                keep_mask[r1] = False
                keep_mask[r2] = False
                for rm in removed:
                    keep_mask[rm] = False
                n_keep = keep_mask.sum()
                if n_keep == 0:
                    continue
                U_test = current_U[:, keep_mask]
                V_test = current_V[:, keep_mask]
                W_test = current_W[:, keep_mask]
                err = verify_fast(U_test, V_test, W_test)
                if err < 1e-7:
                    removed.add(r1)
                    removed.add(r2)
                    current_rank -= 2

    # Pass 4: Another randomized single removal pass
    if time.time() < deadline - 10:
        rng = np.random.default_rng(777)
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            remaining = [r for r in range(rank) if r not in removed]
            if not remaining:
                break
            r = rng.choice(remaining)
            ok, _ = try_remove_term(current_U, current_V, current_W, r, removed)
            if ok:
                removed.add(r)
                current_rank -= 1

    if removed:
        keep = [r for r in range(rank) if r not in removed]
        U_red = current_U[:, keep]
        V_red = current_V[:, keep]
        W_red = current_W[:, keep]
        err_red = verify_fast(U_red, V_red, W_red)
        if err_red < 1e-7:
            U, V, W = U_red, V_red, W_red

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
