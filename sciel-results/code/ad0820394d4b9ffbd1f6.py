def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110

    T_target = np.zeros((n * m, m * p, p * n))
    for i_A in range(n):
        for j in range(m):
            for k_B in range(p):
                T_target[i_A * m + j, j * p + k_B, k_B * n + i_A] = 1.0

    # Try to find lower-rank solutions via alternating minimization
    for target_rank in [38, 39, 40, 41]:
        if time.time() > deadline - 15:
            break
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            np.random.seed(target_rank * 1000 + seed)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)

            for it in range(150):
                if time.time() > deadline - 2:
                    break
                # Update U
                B_flat = np.stack([(V[:, r] * W[:, r]).reshape(-1) for r in range(target_rank)], axis=1)
                T_flat = T_target.reshape(n * m, -1)
                U = T_flat @ np.linalg.pinv(B_flat)

                # Update V
                C_flat = np.stack([(U[:, r] * W[:, r]).reshape(-1) for r in range(target_rank)], axis=1)
                T_flat_V = T_target.transpose(1, 0, 2).reshape(m * p, -1)
                V = T_flat_V @ np.linalg.pinv(C_flat)

                # Update W
                D_flat = np.stack([(U[:, r] * V[:, r]).reshape(-1) for r in range(target_rank)], axis=1)
                T_flat_W = T_target.transpose(2, 0, 1).reshape(p * n, -1)
                W = T_flat_W @ np.linalg.pinv(D_flat)

            # Check rounded solution
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            T_comp = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            if np.allclose(T_comp, T_target, atol=1e-9):
                return {
                    "U": [[float(x) for x in row] for row in U_r],
                    "V": [[float(x) for x in row] for row in V_r],
                    "W": [[float(x) for x in row] for row in W_r]
                }

    # Fallback: Strassen rank-42
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    U_strassen = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]

    V_strassen = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]

    W_strassen = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (3, 1)],
        [(0, 1), (2, 1)],
        [(0, -1), (1, 1)],
        [(3, 1)],
        [(0, 1)]
    ]

    col = 0
    for b in range(3):
        for a in range(2):
            for s in range(7):
                for (idx, coeff) in U_strassen[s]:
                    i_loc = idx // 2
                    j_loc = idx % 2
                    U[i_loc * m + a * 2 + j_loc, col] += coeff
                for (idx, coeff) in V_strassen[s]:
                    j_loc = idx // 2
                    k_loc = idx % 2
                    V[(a * 2 + j_loc) * p + b * 2 + k_loc, col] += coeff
                for (idx, coeff) in W_strassen[s]:
                    i_loc = idx // 2
                    k_loc = idx % 2
                    W[(b * 2 + k_loc) * n + i_loc, col] += coeff
                col += 1

    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
