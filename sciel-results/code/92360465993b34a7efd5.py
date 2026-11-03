import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n

    def time_left():
        return DEADLINE_S - (time.time() - start_time)

    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def validate(U, V, W, tol=1e-10):
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        T_expected = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i,j,k] = 1.0

        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            u_mat = u_c.reshape(n, m)
            v_mat = v_c.reshape(m, p)
            w_mat = w_c.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i,j]) < 1e-15:
                        continue
                    for k in range(p):
                        T_computed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]

        return np.max(np.abs(T_computed - T_expected)) < tol

    # Strassen coefficients for 2x2x2
    U_coeffs = [
        {0: 1, 3: 1},       # M1: a, d
        {2: 1, 3: 1},       # M2: c, d
        {0: 1},             # M3: a
        {3: 1},             # M4: d
        {0: 1, 1: 1},       # M5: a, b
        {2: 1, 0: -1},      # M6: c, -a
        {1: 1, 3: -1}       # M7: b, -d
    ]

    V_coeffs = [
        {0: 1, 3: 1},       # M1: e, h
        {0: 1},             # M2: e
        {1: 1, 3: -1},      # M3: f, -h
        {2: 1, 0: -1},      # M4: g, -e
        {3: 1},             # M5: h
        {0: 1, 1: 1},       # M6: e, f
        {2: 1, 3: 1}        # M7: g, h
    ]

    W_coeffs = [
        {0: 1, 3: 1},       # M1: C00, C11
        {2: 1, 3: -1},      # M2: C10, -C11
        {1: 1, 3: 1},       # M3: C01, C11
        {0: 1, 2: 1},       # M4: C00, C10
        {0: -1, 1: 1},      # M5: -C00, C01
        {3: 1},             # M6: C11
        {0: 1}              # M7: C00
    ]

    def build_block_decomposition(blocks):
        """
        Build a decomposition with multiple non-overlapping Strassen blocks.
        blocks: list of (i_off, j_off, k_off) tuples
        """
        # Determine covered terms
        covered = set()
        for (i_off, j_off, k_off) in blocks:
            for i in range(i_off, i_off+2):
                for j in range(j_off, j_off+2):
                    for k in range(k_off, k_off+2):
                        covered.add((i,j,k))

        total_covered = len(covered)
        schoolbook_count = n*m*p - total_covered
        rank = 7*len(blocks) + schoolbook_count

        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        col = 0

        # Strassen blocks
        for (i_off, j_off, k_off) in blocks:
            for l in range(7):
                for li, coeff in U_coeffs[l].items():
                    i_local = li // 2
                    j_local = li % 2
                    i_global = i_off + i_local
                    j_global = j_off + j_local
                    U[i_global*m + j_global, col] += coeff
                for lj, coeff in V_coeffs[l].items():
                    j_local = lj // 2
                    k_local = lj % 2
                    j_global = j_off + j_local
                    k_global = k_off + k_local
                    V[j_global*p + k_global, col] += coeff
                for lc, coeff in W_coeffs[l].items():
                    i_local = lc // 2
                    k_local = lc % 2
                    i_global = i_off + i_local
                    k_global = k_off + k_local
                    W[k_global*n + i_global, col] += coeff
                col += 1

        # Schoolbook for remaining
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i,j,k) in covered:
                        continue
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1

        return {"U": U, "V": V, "W": W}

    def find_max_disjoint_blocks():
        """
        Find the maximum number of disjoint 2x2x2 Strassen blocks.
        Two blocks are disjoint if they don't overlap in all three dimensions.
        Actually, for a valid decomposition, blocks must not share any (i,j,k) term.
        So two blocks (i1,j1,k1) and (i2,j2,k2) are disjoint iff
        they don't share any i, OR don't share any j, OR don't share any k.
        """
        all_blocks = []
        for i_off in range(n-1):
            for j_off in range(m-1):
                for k_off in range(p-1):
                    all_blocks.append((i_off, j_off, k_off))

        def blocks_disjoint(b1, b2):
            i1, j1, k1 = b1
            i2, j2, k2 = b2
            # Check if they share any (i,j,k)
            share_i = not (i1+1 < i2 or i2+1 < i1)
            share_j = not (j1+1 < j2 or j2+1 < j1)
            share_k = not (k1+1 < k2 or k2+1 < k1)
            # They are NOT disjoint if they share all three dimensions
            return not (share_i and share_j and share_k)

        # Greedy: try to pack as many blocks as possible
        best_set = []
        for b in all_blocks:
            if all(blocks_disjoint(b, existing) for existing in best_set):
                best_set.append(b)

        return best_set

    best_result = None
    best_rank = n*m*p  # schoolbook

    # Try single blocks
    for i_off in range(n-1):
        for j_off in range(m-1):
            for k_off in range(p-1):
                if time_left() < 10:
                    break
                result = build_block_decomposition([(i_off, j_off, k_off)])
                if validate(result["U"], result["V"], result["W"]):
                    rank = result["U"].shape[1]
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {
                            "U": result["U"].tolist(),
                            "V": result["V"].tolist(),
                            "W": result["W"].tolist()
                        }

    # Try pairs of blocks
    all_blocks = []
    for i_off in range(n-1):
        for j_off in range(m-1):
            for k_off in range(p-1):
                all_blocks.append((i_off, j_off, k_off))

    def blocks_disjoint(b1, b2):
        i1, j1, k1 = b1
        i2, j2, k2 = b2
        share_i = not (i1+1 < i2 or i2+1 < i1)
        share_j = not (j1+1 < j2 or j2+1 < j1)
        share_k = not (k1+1 < k2 or k2+1 < k1)
        return not (share_i and share_j and share_k)

    for idx1, b1 in enumerate(all_blocks):
        for idx2, b2 in enumerate(all_blocks):
            if idx2 <= idx1:
                continue
            if time_left() < 5:
                break
            if not blocks_disjoint(b1, b2):
                continue
            result = build_block_decomposition([b1, b2])
            if validate(result["U"], result["V"], result["W"]):
                rank = result["U"].shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_result = {
                        "U": result["U"].tolist(),
                        "V": result["V"].tolist(),
                        "W": result["W"].tolist()
                    }

    # Try triples of blocks
    for idx1, b1 in enumerate(all_blocks):
        for idx2, b2 in enumerate(all_blocks):
            if idx2 <= idx1:
                continue
            if not blocks_disjoint(b1, b2):
                continue
            for idx3, b3 in enumerate(all_blocks):
                if idx3 <= idx2:
                    continue
                if time_left() < 2:
                    break
                if not blocks_disjoint(b1, b3) or not blocks_disjoint(b2, b3):
                    continue
                result = build_block_decomposition([b1, b2, b3])
                if validate(result["U"], result["V"], result["W"]):
                    rank = result["U"].shape[1]
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {
                            "U": result["U"].tolist(),
                            "V": result["V"].tolist(),
                            "W": result["W"].tolist()
                        }

    # Try the greedy max set
    greedy_blocks = find_max_disjoint_blocks()
    if len(greedy_blocks) >= 2:
        if time_left() > 5:
            result = build_block_decomposition(greedy_blocks)
            if validate(result["U"], result["V"], result["W"]):
                rank = result["U"].shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_result = {
                        "U": result["U"].tolist(),
                        "V": result["V"].tolist(),
                        "W": result["W"].tolist()
                    }

    if best_result is not None:
        return best_result
    else:
        return schoolbook()
