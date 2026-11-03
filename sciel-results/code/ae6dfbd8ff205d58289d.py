<<<<<<< SEARCH
    def randomized_search(U, V, W, rank, iterations=1000):
        """Randomly perturb the decomposition and try to reduce rank."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        T_true = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i * m + j, j * p + k, k * n + i] = 1.0

        best_U, best_V, best_W, best_rank = U, V, W, rank

        for it in range(iterations):
            if time.time() - start_time > DEADLINE:
                break

            # Randomly perturb
            U_pert = U_np.copy()
            V_pert = V_np.copy()
            W_pert = W_np.copy()

            # Small random perturbations
            noise_scale = 0.1
            U_pert += np.random.randn(n * m, rank) * noise_scale
            V_pert += np.random.randn(m * p, rank) * noise_scale
            W_pert += np.random.randn(p * n, rank) * noise_scale

            # Check if still valid (should be close but not exact)
            # Try to project back to integer/half-integer
            U_round = np.round(U_pert * 2) / 2
            V_round = np.round(V_pert * 2) / 2
            W_round = np.round(W_pert * 2) / 2

            T_test = np.zeros((n * m, m * p, p * n))
            for c in range(rank):
                T_test += np.einsum('i,j,k->ijk', U_round[:, c], V_round[:, c], W_round[:, c])

            if np.allclose(T_test, T_true, atol=1e-10):
                # Valid! Try to reduce rank
                U_new, V_new, W_new, new_rank = try_remove_factors(
                    [[U_round[i, c] for c in range(rank)] for i in range(n * m)],
                    [[V_round[i, c] for c in range(rank)] for i in range(m * p)],
                    [[W_round[i, c] for c in range(rank)] for i in range(p * n)],
                    new_rank
                )
                if new_rank < best_rank:
                    best_U, best_V, best_W, best_rank = U_new, V_new, W_new, new_rank

        return best_U, best_V, best_W, best_rank
=======
    def iht_search(U, V, W, rank, max_iter=200):
        """Iterative Hard Thresholding approach to find lower rank decomposition."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Build true tensor
        T_true = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i * m + j, j * p + k, k * n + i] = 1.0

        best_U, best_V, best_W, best_rank = U, V, W, rank

        # Try different target ranks below current
        for target_rank in range(max(10, rank - 15), rank):
            if time.time() - start_time > DEADLINE - 5:
                break

            # Initialize with random factors
            for init_iter in range(3):
                if time.time() - start_time > DEADLINE - 5:
                    break

                # Random initialization
                U_cand = np.random.randn(n * m, target_rank)
                V_cand = np.random.randn(m * p, target_rank)
                W_cand = np.random.randn(p * n, target_rank)

                # IHT iterations
                for it in range(max_iter):
                    if time.time() - start_time > DEADLINE - 5:
                        break

                    # Compute current tensor
                    T_cur = np.zeros((n * m, m * p, p * n))
                    for c in range(target_rank):
                        T_cur += np.einsum('i,j,k->ijk', U_cand[:, c], V_cand[:, c], W_cand[:, c])

                    # Check if close to target
                    err = np.linalg.norm(T_cur - T_true) / np.linalg.norm(T_true)
                    if err < 1e-8:
                        # Valid decomposition found
                        U_list = [[U_cand[i, c] for c in range(target_rank)] for i in range(n * m)]
                        V_list = [[V_cand[i, c] for c in range(target_rank)] for i in range(m * p)]
                        W_list = [[W_cand[i, c] for c in range(target_rank)] for i in range(p * n)]
                        if target_rank < best_rank:
                            best_U, best_V, best_W, best_rank = U_list, V_list, W_list, target_rank
                        break

                    # Update factors using gradient descent
                    # Gradient w.r.t. U: sum_c W_c * V_c^T (reshaped)
                    grad_U = np.zeros((n * m, target_rank))
                    grad_V = np.zeros((m * p, target_rank))
                    grad_W = np.zeros((p * n, target_rank))

                    for c in range(target_rank):
                        # d/dU_c = (T_true - T_cur) contracted with V_c and W_c
                        # This is complex; use simple projection instead
                        pass

                    # Simple approach: project to nearest valid decomposition
                    # For each factor, try to adjust to reduce error
                    step_size = 0.01
                    for c in range(target_rank):
                        # Compute contribution of factor c
                        contrib = np.einsum('i,j,k->ijk', U_cand[:, c], V_cand[:, c], W_cand[:, c])
                        diff = T_true - T_cur
                        # Approximate gradient direction
                        U_cand[:, c] += step_size * np.sum(diff * np.einsum('j,k->ijk', V_cand[:, c], W_cand[:, c]), axis=(1, 2))
                        V_cand[:, c] += step_size * np.sum(diff * np.einsum('i,k->ijk', U_cand[:, c], W_cand[:, c]), axis=(0, 2))
                        W_cand[:, c] += step_size * np.sum(diff * np.einsum('i,j->ijk', U_cand[:, c], V_cand[:, c]), axis=(0, 1))

        return best_U, best_V, best_W, best_rank
>>>>>>> REPLACE
