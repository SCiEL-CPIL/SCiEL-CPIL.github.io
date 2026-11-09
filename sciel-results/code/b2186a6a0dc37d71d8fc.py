import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def _evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Random search for initial diversity (first 10% of budget)
        n_random = max(5, budget // 10)
        for _ in range(n_random):
            if calls >= budget or time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            _evaluate(x)

        # Phase 2: CMA-ES with restarts
        # We'll do multiple restarts, each with a fraction of the remaining budget
        n_restarts = 5
        remaining = budget - calls
        if remaining <= 0:
            return best_x, best_f

        # Allocate budget: first restart gets more, subsequent get less
        # Front-loaded: 40%, 25%, 20%, 10%, 5%
        fractions = [0.40, 0.25, 0.20, 0.10, 0.05]
        allocations = [int(remaining * f) for f in fractions]
        # Ensure last restart gets remainder
        allocations[-1] = remaining - sum(allocations[:-1])
        allocations = [max(10, a) for a in allocations]

        for r in range(n_restarts):
            if calls >= budget or time.time() >= deadline:
                break

            alloc = allocations[r]
            
            # Initialize CMA-ES parameters
            if r == 0:
                m = best_x.copy()
                sigma = 1.0
            else:
                m = rng.uniform(-4, 4, dim)
                sigma = 2.0

            # CMA-ES parameters
            lam = 12
            mu = lam // 2
            w = np.log(np.arange(1, mu + 1) + 0.5) - np.log(mu + 0.5)
            w = w / np.sum(w)
            mu_eff = np.sum(w) ** 2 / np.sum(w ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = 3.0 / (dim + 4)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            steps_in_restart = 0
            max_steps_in_restart = alloc // lam + 10

            while steps_in_restart < max_steps_in_restart and calls < budget:
                if time.time() >= deadline:
                    break

                # Sample population
                Z = rng.standard_normal((lam, dim))
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

                X = m + sigma * (Z @ C_sqrt.T)
                X = np.clip(X, -5, 5)

                # Evaluate
                F = np.array([_evaluate(x) for x in X])

                # Sort
                idx_sort = np.argsort(F)
                X_sorted = X[idx_sort]
                F_sorted = F[idx_sort]

                # Update mean
                m_new = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)

                # Update step size
                c_1_mu = c_1 + c_mu
                eigvals_safe = np.clip(eigvals, 1e-12, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals_safe)) @ eigvecs.T

                delta = (m_new - m) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(c_1_mu * (2 - c_1_mu)) * (C_inv_sqrt @ delta)

                chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (25 * dim ** 2))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update covariance
                p_c = (1 - c_c) * p_c + c_c * delta
                Y_sorted = (X_sorted[:mu] - m) / sigma

                C_new = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
                for j in range(mu):
                    C_new += c_mu * w[j] * np.outer(Y_sorted[j], Y_sorted[j])

                C_new = (C_new + C_new.T) / 2
                m = m_new
                C = C_new
                steps_in_restart += 1

                # If sigma becomes too small, stop this restart
                if sigma < 1e-10:
                    break

        return best_x, best_f
