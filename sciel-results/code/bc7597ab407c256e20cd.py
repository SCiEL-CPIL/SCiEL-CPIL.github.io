import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s limit

    # Build the 4x4x4 Strassen decomposition (rank 49) + schoolbook for col 4 (rank 16) = 65 terms
    # Then attempt rank reduction by merging identical rank-1 vectors.

    U_list = []  # list of (i*m+j, coeff) sparse representations
    V_list = []
    W_list = []

    def add_term(u_sparse, v_sparse, w_sparse):
        """u_sparse: list of (idx, coeff), v_sparse: list of (idx, coeff), w_sparse: list of (idx, coeff)"""
        # Convert to dense for the term
        u_dense = [0.0] * (n * m)
        for idx, c in u_sparse:
            if c != 0:
                u_dense[idx] += c
        v_dense = [0.0] * (m * p)
        for idx, c in v_sparse:
            if c != 0:
                v_dense[idx] += c
        w_dense = [0.0] * (p * n)
        for idx, c in w_sparse:
            if c != 0:
                w_dense[idx] += c
        U_list.append(u_dense)
        V_list.append(v_dense)
        W_list.append(w_dense)

    # --- Strassen 4x4x4 decomposition (49 terms) ---
    # Super Strassen terms for 4x4 * 4x4
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)

    super_defs = [
        # (P_coeffs for A blocks [A00,A01,A10,A11], Q_coeffs for B blocks [B00,B01,B10,B11])
        ([1, 0, 0, 1], [1, 0, 0, 1]),   # M1 = (A00+A11)(B00+B11)
        ([0, 0, 1, 1], [1, 0, 0, 0]),   # M2 = (A10+A11)B00
        ([1, 0, 0, 0], [0, 1, 0, -1]),  # M3 = A00(B01-B11)
        ([0, 0, 0, 1], [-1, 0, 1, 0]),  # M4 = A11(B10-B00)
        ([1, 1, 0, 0], [0, 0, 0, 1]),   # M5 = (A00+A01)B11
        ([-1, 0, 1, 0], [1, 1, 0, 0]),  # M6 = (A10-A00)(B00+B01)
        ([0, 1, 0, -1], [0, 0, 1, 1])   # M7 = (A01-A11)(B10+B11)
    ]

    # Inner Strassen terms for 2x2 * 2x2
    # P = [[p00, p01], [p10, p11]], Q = [[q00, q01], [q10, q11]]
    # s1 = (p00+p11)(q00+q11)
    # s2 = (p10+p11)q00
    # s3 = p00(q01-q11)
    # s4 = p11(q10-q00)
    # s5 = (p00+p01)q11
    # s6 = (p10-p00)(q00+q01)
    # s7 = (p01-p11)(q10+q11)
    # S00 = s1 + s4 - s5 + s7
    # S01 = s3 + s5
    # S10 = s2 + s4
    # S11 = s1 - s2 + s3 + s6

    inner_defs = [
        # (p_elems: [(idx, coeff)], q_elems: [(idx, coeff)], s_contrib: [S00, S01, S10, S11])
        # idx: 0=p00, 1=p01, 2=p10, 3=p11
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [1, 0, 0, 1]),    # s1
        ([(2, 1), (3, 1)], [(0, 1)], [0, 0, 1, -1]),            # s2
        ([(0, 1)], [(1, 1), (3, -1)], [0, 1, 0, 1]),            # s3
        ([(3, 1)], [(2, 1), (0, -1)], [1, 0, 1, 0]),            # s4
        ([(0, 1), (1, 1)], [(3, 1)], [-1, 1, 0, 0]),            # s5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [0, 0, 0, 1]),    # s6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [1, 0, 0, 0])     # s7
    ]

    # C block contributions from each super term
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    c_block_contribs = [
        [1, 0, 0, 1],   # M1 -> C00(+1), C11(+1)
        [0, 0, 1, -1],  # M2 -> C10(+1), C11(-1)
        [0, 1, 0, 1],   # M3 -> C01(+1), C11(+1)
        [1, 0, 1, 0],   # M4 -> C00(+1), C10(+1)
        [-1, 1, 0, 0],  # M5 -> C00(-1), C01(+1)
        [0, 0, 0, 1],   # M6 -> C11(+1)
        [1, 0, 0, 0]    # M7 -> C00(+1)
    ]

    def expand_A_block(block_idx, r, c, coeff):
        """Expand a block element to global (i, j, coeff)"""
        if block_idx == 0:
            return (r, c, coeff)
        elif block_idx == 1:
            return (r, 2 + c, coeff)
        elif block_idx == 2:
            return (2 + r, c, coeff)
        elif block_idx == 3:
            return (2 + r, 2 + c, coeff)

    def expand_B_block(block_idx, r, c, coeff):
        """Expand a block element to global (j, k, coeff)"""
        if block_idx == 0:
            return (r, c, coeff)
        elif block_idx == 1:
            return (r, 2 + c, coeff)
        elif block_idx == 2:
            return (2 + r, c, coeff)
        elif block_idx == 3:
            return (2 + r, 2 + c, coeff)

    # Build all 49 terms for 4x4x4
    for s_idx in range(7):
        p_blocks, q_blocks = super_defs[s_idx]
        c_contribs = c_block_contribs[s_idx]

        for t_idx in range(7):
            p_elems, q_elems, s_contrib = inner_defs[t_idx]

            u_sparse = []
            v_sparse = []
            w_sparse = []

            # Build U (A part)
            for p_elem_idx, p_coeff in p_elems:
                r = p_elem_idx // 2
                c = p_elem_idx % 2
                for b_idx, b_coeff in enumerate(p_blocks):
                    if b_coeff == 0:
                        continue
                    i, j, coeff = expand_A_block(b_idx, r, c, b_coeff)
                    total_coeff = coeff * p_coeff
                    if total_coeff != 0:
                        u_sparse.append((i * m + j, total_coeff))

            # Build V (B part)
            for q_elem_idx, q_coeff in q_elems:
                r = q_elem_idx // 2
                c = q_elem_idx % 2
                for b_idx, b_coeff in enumerate(q_blocks):
                    if b_coeff == 0:
                        continue
                    j, k, coeff = expand_B_block(b_idx, r, c, b_coeff)
                    total_coeff = coeff * q_coeff
                    if total_coeff != 0:
                        v_sparse.append((j * p + k, total_coeff))

            # Build W (C part)
            for s_pos in range(4):
                s_coeff = s_contrib[s_pos]
                if s_coeff == 0:
                    continue
                s_r = s_pos // 2
                s_c = s_pos % 2

                for c_b in range(4):
                    c_coeff = c_contribs[c_b]
                    if c_coeff == 0:
                        continue
                    total_coeff = s_coeff * c_coeff
                    if total_coeff == 0:
                        continue

                    # Map to global (i, k)
                    if c_b == 0:   # C00
                        gi = s_r
                        gk = s_c
                    elif c_b == 1: # C01
                        gi = s_r
                        gk = 2 + s_c
                    elif c_b == 2: # C10
                        gi = 2 + s_r
                        gk = s_c
                    elif c_b == 3: # C11
                        gi = 2 + s_r
                        gk = 2 + s_c

                    w_sparse.append((gk * n + gi, total_coeff))

            add_term(u_sparse, v_sparse, w_sparse)

    # --- Schoolbook for column 4 (16 terms) ---
    # C[i, 4] = sum_j A[i, j] * B[j, 4]
    for i in range(n):
        for j in range(m):
            u_sparse = [(i * m + j, 1)]
            v_sparse = [(j * p + 4, 1)]
            w_sparse = [(4 * n + i, 1)]
            add_term(u_sparse, v_sparse, w_sparse)

    # Now we have 65 terms. Attempt rank reduction.
    # Check if time allows
    if time.time() - start_time > DEADLINE - 10:
        # Convert to dense arrays and return
        rank = len(U_list)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for t in range(rank):
            for idx, val in enumerate(U_list[t]):
                if val != 0:
                    U[idx, t] = val
            for idx, val in enumerate(V_list[t]):
                if val != 0:
                    V[idx, t] = val
            for idx, val in enumerate(W_list[t]):
                if val != 0:
                    W[idx, t] = val
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # --- Rank reduction: merge terms with identical (or proportional) factor vectors ---
    # For each pair of terms, check if they can be merged.
    # Two terms t1 and t2 can be merged if:
    #   U[t2] = alpha * U[t1], V[t2] = beta * V[t1], W[t2] = gamma * W[t1]
    #   and alpha * beta * gamma = 1 (so the product is preserved)
    # In practice, we look for exact matches first (alpha=beta=gamma=1).
    
    # Convert to numpy arrays for efficient comparison
    rank = len(U_list)
    U_arr = np.array(U_list, dtype=float).T  # shape (n*m, rank)
    V_arr = np.array(V_list, dtype=float).T  # shape (m*p, rank)
    W_arr = np.array(W_list, dtype=float).T  # shape (p*n, rank)

    # Try to find and merge identical terms
    # For efficiency, use a hash-based approach
    merged_U = []
    merged_V = []
    merged_W = []
    used = [False] * rank

    # Group terms by their U vector (rounded to handle floating point)
    from collections import defaultdict
    u_groups = defaultdict(list)
    for t in range(rank):
        # Round to 6 decimal places for hashing
        u_key = tuple(np.round(U_arr[:, t], 6))
        u_groups[u_key].append(t)

    for u_key, terms in u_groups.items():
        if len(terms) <= 1:
            continue
        # For terms with same U, check if V and W are also identical
        # Group further by V
        v_groups = defaultdict(list)
        for t in terms:
            v_key = tuple(np.round(V_arr[:, t], 6))
            v_groups[v_key].append(t)

        for v_key, v_terms in v_groups.items():
            if len(v_terms) <= 1:
                continue
            # Group by W
            w_groups = defaultdict(list)
            for t in v_terms:
                w_key = tuple(np.round(W_arr[:, t], 6))
                w_groups[w_key].append(t)

            for w_key, w_terms in w_groups.items():
                if len(w_terms) <= 1:
                    continue
                # These terms are identical - merge them by summing
                # Actually, if they're identical rank-1 terms, the sum is just the sum of coefficients
                # But in our construction, each term has its own coefficient embedded in U, V, W
                # So merging means: new_U = sum of U[t] for t in w_terms, etc.
                # But that's not right either. If U1=U2, V1=V2, W1=W2, then:
                #   U1*V1*W1 + U2*V2*W2 = 2*U1*V1*W1
                # So we can just double the coefficients.
                # In our representation, the coefficients are already in the vectors.
                # So we sum the vectors.
                if not used[w_terms[0]]:
                    new_U = np.zeros(n * m)
                    new_V = np.zeros(m * p)
                    new_W = np.zeros(p * n)
                    for t in w_terms:
                        if not used[t]:
                            new_U += U_arr[:, t]
                            new_V += V_arr[:, t]
                            new_W += W_arr[:, t]
                            used[t] = True
                    merged_U.append(new_U)
                    merged_V.append(new_V)
                    merged_W.append(new_W)

    # Add non-merged terms
    for t in range(rank):
        if not used[t]:
            merged_U.append(U_arr[:, t])
            merged_V.append(V_arr[:, t])
            merged_W.append(W_arr[:, t])

    # If no merging happened, try proportional merging
    # This is more complex; skip for now if time is tight
    if time.time() - start_time > DEADLINE - 5:
        pass

    new_rank = len(merged_U)
    U_final = np.array(merged_U, dtype=float)
    V_final = np.array(merged_V, dtype=float)
    W_final = np.array(merged_W, dtype=float)

    # Round to clean values (should be integers or half-integers)
    U_final = np.round(U_final, 10)
    V_final = np.round(V_final, 10)
    W_final = np.round(W_final, 10)

    # Clean up tiny values
    U_final[np.abs(U_final) < 1e-9] = 0
    V_final[np.abs(V_final) < 1e-9] = 0
    W_final[np.abs(W_final) < 1e-9] = 0

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
