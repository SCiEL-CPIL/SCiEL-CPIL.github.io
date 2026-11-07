"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        # Wrap func to count calls and clip
        call_count = [0]

        def wrapped(x):
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            call_count[0] += 1
            return float(func(x))

        best_x = None
        best_f = np.inf

        def cma_es(x0, sigma, remaining_budget):
            nonlocal best_x, best_f, call_count
            n = dim
            # CMA-ES parameters
            mu = max(1, n // 2)  # number of selected individuals
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / ((n + 1.3) ** 2 + mueff)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # State
            mean = np.array(x0, dtype=np.float64)
            sigma = float(sigma)
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            C_inv = np.eye(n)
            # Precompute sqrt of C via Cholesky
            try:
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(B @ B.T))
            except:
                B = np.eye(n)
                D = np.ones(n)

            pop_size = 2 * mu
            calls_in_run = 0
            no_improve = 0
            best_in_run = wrapped(mean)
            calls_in_run += 1

            for gen in range(remaining_budget // pop_size + 1):
                if time.time() > deadline - 5.0:
                    break
                if call_count[0] >= budget - 2:
                    break

                # Generate population
                Y = rng.standard_normal((pop_size, n))
                # Transform: mean + sigma * B @ (D * Y)  -- approximate
                # Actually, for CMA-ES we need B such that C = B @ diag(D^2) @ B.T
                # Use eigendecomposition for simplicity
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    D_sqrt = np.sqrt(eigvals)
                    # Y @ (D_sqrt[:, None] * eigvecs.T)
                    X = mean[None, :] + sigma * (Y * D_sqrt[None, :]) @ eigvecs.T
                except:
                    X = mean[None, :] + sigma * Y

                X = np.clip(X, -5.0, 5.0)

                # Evaluate
                F = np.array([wrapped(xi) for xi in X])
                calls_in_run += pop_size

                # Sort
                idx = np.argsort(F)
                X_selected = X[idx[:mu]]
                F_selected = F[idx[:mu]]

                # Update best
                if F_selected[0] < best_f:
                    best_f = F_selected[0]
                    best_x = X_selected[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Update mean
                new_mean = np.sum(weights[:, None] * X_selected, axis=0)
                new_mean = np.clip(new_mean, -5.0, 5.0)

                # Update evolution paths
                diff_mean = (new_mean - mean) / sigma
                pc = (1 - c_sigma) * pc + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * diff_mean
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * \
                     np.linalg.inv(eigvecs @ np.diag(D_sqrt) @ eigvecs.T).dot(diff_mean) if False else \
                     (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * diff_mean

                # Update covariance matrix
                delta_d = (1 + c_sigma * (2 - c_sigma)) * (np.linalg.norm(ps) ** 2 / n - 1)
                rank_one = np.outer(ps, ps)
                rank_mu = np.sum(weights[:, None, None] * X_selected[:, :, None] * X_selected[:, None, :], axis=0)
                C = (1 + c_1 * delta_d - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2.0
                try:
                    eigvals_c, _ = np.linalg.eigh(C)
                    if np.any(eigvals_c < 1e-20):
                        C += 1e-10 * np.eye(n)
                except:
                    pass

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                mean = new_mean

                # Check for stagnation
                if no_improve > 50:
                    break

                if call_count[0] >= budget - 2:
                    break

            return best_x, best_f

        rng = np.random.default_rng(42)

        # Strategy: Use CMA-ES with restarts
        # Phase 1: A few random starts to find a good region
        # Phase 2: CMA-ES from best point
        # Phase 3: If budget remains, restart from perturbed best

        # Initial random evaluations to find a starting point
        n_init = min(20, budget // 10)
        for i in range(n_init):
            if time.time() > deadline - 10.0:
                break
            if call_count[0] >= budget - 5:
                break
            x = rng.uniform(-5, 5, dim)
            fx = wrapped(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)

        # CMA-ES run 1
        remaining = budget - call_count[0]
        if remaining > 2 * dim:
            x0 = best_x.copy()
            sigma0 = 1.0
            cma_es(x0, sigma0, remaining)

        # CMA-ES run 2 (restart with smaller sigma)
        remaining = budget - call_count[0]
        if remaining > 2 * dim and time.time() < deadline - 10.0:
            x0 = best_x.copy()
            sigma0 = 0.3
            cma_es(x0, sigma0, remaining)

        # CMA-ES run 3 (perturb and restart)
        remaining = budget - call_count[0]
        if remaining > 2 * dim and time.time() < deadline - 10.0:
            x0 = best_x + 0.5 * rng.standard_normal(dim)
            x0 = np.clip(x0, -5, 5)
            sigma0 = 0.5
            cma_es(x0, sigma0, remaining)

        return best_x, best_f
# EVOLVE-BLOCK-END
