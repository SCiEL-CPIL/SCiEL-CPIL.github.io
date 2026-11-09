"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Track best overall
        best_x = np.zeros(n)
        best_f = np.inf
        call_count = 0

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))  # population size
        mu = lambda_ // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 - c_c + mu_eff / (n + 2)) / ((n + 2) + c_c + mu_eff / (n + 2))
        c_mu = (mu_eff - 2 + 1 / c_1) / ((n + 2) + c_c + mu_eff - 2 + 1 / c_1)
        c_mu = min(c_mu, 1 - c_1)

        def cma_es(x0, f0, sigma, calls_used, remaining_budget):
            nonlocal best_x, best_f, call_count, deadline

            x = x0.copy()
            fx = f0
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            B = np.eye(n)
            D = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            step_size = sigma

            # Precompute for efficiency
            for it in range(remaining_budget):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                ys = np.array([step_size * rng.standard_normal(n) for _ in range(lambda_)])
                zs = (ys @ B.T) / D  # Apply covariance
                # Actually, let's do it properly:
                # ys are standard normals, we need y = D * B * z where z ~ N(0,I)
                # But we generate z ~ N(0,I), then y = D @ (B @ z)
                # Wait, the standard CMA-ES: y_i = D @ (B @ z_i) where z_i ~ N(0,I)
                # Then x_i = x + y_i
                # Let me redo:
                zs = rng.standard_normal((lambda_, n))
                ys = (D @ (B.T @ zs.T)).T  # shape (lambda_, n)
                # Actually: y_i = D @ B @ z_i, so for each i: y_i = D @ (B @ z_i)
                # In matrix form: Y = (D @ B @ Z.T).T where Z is (lambda_, n)
                # Let me just do it element-wise for clarity and correctness
                ys = np.zeros((lambda_, n))
                for i in range(lambda_):
                    ys[i] = D @ (B @ zs[i])
                # Clip to bounds
                xs = x + ys
                for i in range(n):
                    xs[:, i] = np.clip(xs[:, i], -5, 5)

                # Evaluate
                fs = np.zeros(lambda_)
                for i in range(lambda_):
                    if call_count >= budget:
                        fs[i] = np.inf
                    else:
                        fs[i] = func(xs[i])
                        call_count += 1
                        if fs[i] < best_f:
                            best_f = fs[i]
                            best_x = xs[i].copy()

                # Sort
                idx = np.argsort(fs)
                xs = xs[idx]
                fs = fs[idx]

                # Update mean
                w = weights[:mu]
                x_new = (xs[:mu] * w[:, np.newaxis]).sum(axis=0)

                # Update evolution paths
                sigma_old = step_size
                if sigma_old > 1e-12:
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                        B.T @ (D.T @ ((x_new - x) / sigma_old).T)).T
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x)

                    # Update covariance
                    delta_hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1)))
                    h_sigma = 1.0 if delta_hs < (1 + 1.5 / n) * np.sqrt(1 - (1 - c_sigma) ** 2) else 0.0

                    delta_hs_c = np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (2 * (it + 1)))
                    h_c = 1.0 if delta_hs_c < (1 + 2 / n) * np.sqrt(1 - (1 - c_c) ** 2) else 0.0

                    # Update C = D^2 * B B^T
                    diff = (x_new - x) / sigma_old
                    rank_mu_term = (np.outer(diff, diff) * c_mu).sum() if False else np.outer(diff, diff)
                    # Actually: C += c_1*(1 - d_1)*C + c_mu * sum w_i (y_i - x_new)(y_i - x_new)^T / sigma^2
                    # But we're tracking B, D separately, so let's do the standard update
                    # C = D @ D.T * B @ B.T
                    # Let me just track C directly for simplicity
                    pass

                # This is getting complex. Let me use a simpler but effective approach.
                # Break out and use a cleaner implementation below.
                break

            return x, fx

        # Simpler, more robust approach: use a well-tested CMA-ES implementation
        # Let me write a clean one

        def cma_es_clean(x0, f0, sigma0, remaining):
            nonlocal best_x, best_f, call_count, deadline

            x = x0.copy()
            fx = f0
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)

            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = (2 - c_c + mueff / (n + 2)) / ((n + 2) + c_c + mueff / (n + 2))
            c_mu = min((mueff - 2 + 1 / c_1) / ((n + 2) + c_c + mueff - 2 + 1 / c_1), 1 - c_1)

            C = np.eye(n)
            B, D = np.linalg.eigh(C)
            # Ensure D is positive
            D = np.sqrt(np.maximum(D, 1e-12))

            pc = np.zeros(n)
            ps = np.zeros(n)
            sigma = sigma0
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            for it in range(remaining):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate and evaluate population
                zs = rng.standard_normal((lam, n))
                # y_i = sigma * D @ B @ z_i
                ys = (zs @ B.T * D) * sigma
                xs = x + ys
                for i in range(n):
                    xs[:, i] = np.clip(xs[:, i], -5, 5)

                fs = np.full(lam, np.inf)
                for i in range(lam):
                    if call_count >= budget:
                        break
                    fs[i] = func(xs[i])
                    call_count += 1
                    if fs[i] < best_f:
                        best_f = fs[i]
                        best_x = xs[i].copy()

                # Sort
                idx = np.argsort(fs)
                xs = xs[idx]
                fs = fs[idx]

                # Update mean
                x_new = (xs[:mu] * w[:, np.newaxis]).sum(axis=0)

                # Update evolution paths
                if sigma > 1e-12:
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (
                        (B.T @ (D.T @ ((x_new - x) / sigma).T))).T
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x)

                    # Update sigma
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                    sigma = np.clip(sigma, 1e-8, 1e8)

                    # Update C
                    delta_hs = np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (2 * (it + 1)))
                    h_c = 1.0 if delta_hs < (1 + 2 / n) * np.sqrt(1 - (1 - c_c) ** 2) else 0.0

                    C = (1 + c_1 * (h_c - 1) - c_1) * C
                    C += c_1 * (np.outer(pc, pc) - (1 - h_c) * np.eye(n))
                    C += c_mu / sigma ** 2 * np.sum([w[i] * np.outer(xs[i] - x, xs[i] - x) for i in range(mu)], axis=0)

                    # Symmetrize
                    C = (C + C.T) / 2

                    # Eigen decomposition
                    D, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D, 1e-12))

                x = x_new
                fx = fs[0]
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Convergence check
                if sigma < 1e-10 or np.all(np.abs(x - best_x) < 1e-10):
                    break

            return x, fx

        # Multi-start strategy: try a few different initializations
        # First, do a coarse search with several random starts
        n_starts = 5
        starts = []
        for s in range(n_starts):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break
            x0 = rng.uniform(-4, 4, n)
            f0 = func(x0)
            call_count += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            starts.append((f0, x0.copy()))

        # Sort starts by fitness
        starts.sort(key=lambda t: t[0])

        # Run CMA-ES from the best start, then from others if time permits
        for f0, x0 in starts:
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            remaining = budget - call_count
            if remaining < 10:
                break

            # Use a reasonable initial sigma
            sigma0 = 1.5

            cma_es_clean(x0, f0, sigma0, remaining)

            # If we haven't used all budget and time permits, try another start
            if call_count < budget and time.time() < deadline - 30:
                # Try a second CMA-ES from a different start
                for f1, x1 in starts[1:]:
                    if call_count >= budget:
                        break
                    if time.time() > deadline:
                        break
                    remaining2 = budget - call_count
                    if remaining2 < 10:
                        break
                    cma_es_clean(x1, f1, 1.0, remaining2)
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
