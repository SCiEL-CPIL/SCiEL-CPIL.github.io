    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit

        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        c1 = (2 / ((dim + 1.7)**2 + 10))
        cmu = min(1 - c1, 2 * ((mueff + 2) / (dim + 2))**2 - 1)
        cs = (mueff + 1.7) / (dim + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        cc = 4 / (dim + 4)

        # Initialization
        rng = np.random.default_rng(42)
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_made = 1

        # Calculate max generations to stay within budget
        max_gens = (budget - 1) // lambda_

        for gen in range(max_gens):
            if time.time() > deadline:
                break

            # Generate population
            Y = rng.standard_normal((lambda_, dim))
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
            
            X = x_mean + sigma * (Y @ L.T)
            
            # Evaluate population
            F = np.zeros(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                F[i] = func(X[i])
                calls_made += 1
            
            # If we broke early due to time, we might not have all F values
            # But the loop below assumes F is full.
            # We should check if we have enough calls.
            # Actually, if we break the inner loop, we should break the outer loop too.
            if calls_made >= budget:
                break

            # Sort by fitness
            sorted_indices = np.argsort(F)
            X_sorted = X[sorted_indices]
            F_sorted = F[sorted_indices]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()

            # Compute weighted mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * X_sorted[i]
            
            # Compute C_inv_sqrt
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except:
                C_inv_sqrt = np.eye(dim)

            # Step size control (ps update)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ (x_new - x_mean)) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n

            # Mean update
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x_new - x_mean) / sigma
            x_mean = x_mean + cc * pc

            # Covariance update
            delta_sig = np.linalg.norm(ps) / chi_n - 1
            sigma *= np.exp(cs / damps * (delta_sig - 1))
            
            # Rank-1 update
            C = (1 - c1 - cmu) * C
            C += (c1 + hsig * c1 * (2 - cc)) * np.outer(pc, pc)
            
            # Rank-mu update
            if cmu > 0:
                Y_diff = (X_sorted[:mu] - x_mean) / sigma
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(Y_diff[i], Y_diff[i])
                C += cmu * rank_mu

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            eigvals_check, _ = np.linalg.eigh(C)
            if np.any(eigvals_check < 1e-10):
                C += 1e-10 * np.eye(dim)

        return best_x, best_f
