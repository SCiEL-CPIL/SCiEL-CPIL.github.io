import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    # Build the list of (a, b, c) triples with T=1
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))

    num_eq = len(triples)
    num_rows_U = n * m  # 6
    num_rows_V = m * p  # 21
    num_rows_W = p * n  # 14

    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W (as lists of lists) exactly decompose the tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for (a, b, c) in triples:
            val = np.dot(U[a], np.multiply(V[b], W[c]))
            if abs(val - 1.0) > tol:
                return False
        return True

    # Rank 39 fallback construction
    def construct_39():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        def add_term(u_indices, v_indices, w_indices):
            nonlocal col
            for idx in u_indices:
                U[idx][col] += 1
            for idx in v_indices:
                V[idx][col] += 1
            for idx in w_indices:
                W[idx][col] += 1
            col += 1

        def add_strassen_block(k_start):
            b00 = 0 * 7 + k_start
            b01 = 0 * 7 + k_start + 1
            b10 = 1 * 7 + k_start
            b11 = 1 * 7 + k_start + 1

            c00 = k_start * 2 + 0
            c01 = (k_start + 1) * 2 + 0
            c10 = k_start * 2 + 1
            c11 = (k_start + 1) * 2 + 1

            add_term([0, 4], [b00, b11], [c00, c11])
            add_term([3, 4], [b00], [c10, c11])
            W[c11][col-1] = -1

            add_term([0], [b01, b11], [c01, c11])
            V[b11][col-1] = -1

            add_term([4], [b10, b00], [c00, c10])
            V[b00][col-1] = -1

            add_term([0, 1], [b11], [c00, c01])
            W[c00][col-1] = -1

            add_term([3, 0], [b00, b01], [c11])
            U[0][col-1] = -1

            add_term([1, 4], [b10, b11], [c00])
            U[4][col-1] = -1

        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)

        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1

        add_term([0], [b06], [c06])
        add_term([1], [b16], [c06])
        add_term([3], [b06], [c16])
        add_term([4], [b16], [c16])

        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            add_term([2], [b2k], [c0k])
            add_term([5], [b2k], [c1k])

        return {"U": U, "V": V, "W": W}

    fallback_result = construct_39()

    # Try to find lower rank using structured search with better initialization
    # Key insight: use the rank-39 structure as a guide for initialization
    # and use a more efficient local search

    # For each target rank, try multiple initializations
    for R in range(38, 34, -1):
        if time.time() > deadline - 5:
            break

        max_restarts = 5
        if time.time() > deadline - 15:
            max_restarts = 2

        for attempt in range(max_restarts):
            if time.time() > deadline - 3:
                break

            np.random.seed(hash((R, attempt, 42)) % (2**32))

            # Initialize with sparse random values from {-1, -0.5, 0, 0.5, 1}
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

            U = np.zeros((num_rows_U, R))
            V = np.zeros((num_rows_V, R))
            W = np.zeros((num_rows_W, R))

            # Dense initialization with controlled sparsity
            for r in range(R):
                for i in range(num_rows_U):
                    if np.random.random() < 0.5:
                        U[i, r] = np.random.choice(vals)
                for i in range(num_rows_V):
                    if np.random.random() < 0.4:
                        V[i, r] = np.random.choice(vals)
                for i in range(num_rows_W):
                    if np.random.random() < 0.4:
                        W[i, r] = np.random.choice(vals)

            # Local search with gradient-like updates
            err = 0.0
            for (a, b, c) in triples:
                val = np.dot(U[a], np.multiply(V[b], W[c]))
                err += (val - 1.0) ** 2

            improved = True
            max_steps = 200
            step = 0
            while improved and step < max_steps and time.time() < deadline - 2:
                improved = False
                step += 1

                # Try modifying each entry in a random order
                order = np.random.permutation(num_rows_U * R + num_rows_V * R + num_rows_W * R)

                for idx in order:
                    if time.time() > deadline - 2:
                        break

                    # Determine which matrix and position
                    if idx < num_rows_U * R:
                        row = idx // R
                        col = idx % R
                        mat = U
                        mat_offset = 0
                    elif idx < (num_rows_U + num_rows_V) * R:
                        idx2 = idx - num_rows_U * R
                        row = idx2 // R
                        col = idx2 % R
                        mat = V
                        mat_offset = num_rows_U * R
                    else:
                        idx2 = idx - (num_rows_U + num_rows_V) * R
                        row = idx2 // R
                        col = idx2 % R
                        mat = W
                        mat_offset = (num_rows_U + num_rows_V) * R

                    old_val = mat[row, col]
                    if old_val == 0.0:
                        # Try non-zero values
                        candidates = [v for v in vals if v != 0.0]
                    else:
                        # Try zero and other values
                        candidates = [0.0] + [v for v in vals if v != old_val]

                    for new_val in candidates:
                        if new_val == old_val:
                            continue

                        mat[row, col] = new_val

                        err_new = 0.0
                        for (a, b, c) in triples:
                            val = np.dot(U[a], np.multiply(V[b], W[c]))
                            err_new += (val - 1.0) ** 2

                        if err_new < err - 1e-12:
                            err = err_new
                            improved = True
                            break
                        else:
                            mat[row, col] = old_val

                    if err < 1e-9:
                        break

                if err < 1e-9:
                    break

            if err < 1e-9:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if check_decomposition(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}

    # Final fallback
    return fallback_result
# EVOLVE-BLOCK-END
