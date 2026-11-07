"""Optimizer for BBOB f3 (Rastrigin) with CMA-ES + adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for Rastrigin in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # 20s safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # State: best solution and value across all restarts
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        # CMA-ES parameters
        # Start with moderate spread; Rastrigin landscape in [-5,5]^5
        # Use population size based on dim
        lam = 10  # small pop for fast early progress
        mu = lam // 2  # number of parents
        # Weights (log-sum normalization)
        w = np.log(np.arange(mu, 0, -1))
        w /= w.sum()
        mueff = mu * w[0] / (w ** 2).sum()

        def cma_run(x0, sigma0, remaining_budget):
            nonlocal calls_used, best_x, best_f
            d = dim
            n = lam
            # CMA-ES step size control
            cs = 1 / (sqrt_d + 1) if (sqrt_d := np.sqrt(d)) else 0.5
            ds = 1 + 2 * max(0, (mueff - 1) / (d + 1))
            c1 = (2 + (mueff - 2) / (d + 1)) / ((d + 2) ** 2 + 2)
            cmu = min(1 - c1,
                      2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
            # Covariance matrix
            C = np.eye(d)
            # Inverse square root of C (initially identity)
            C_isqrt = np.eye(d)
            # Evolution path for covariance
            p_c = np.zeros(d)
            # Evolution path for step size
            p_sigma = np.zeros(d)
            sigma = sigma0
            x_mean = x0.copy()
            chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            # Local best
            local_best_x = x0.copy()
            local_best_f = np.inf

            it = 0
            while calls_used < remaining_budget:
                if time.time() > deadline:
                    break
                it += 1
                # Generate population
                Y = rng.standard_normal((n, d))
                # Transform: x_mean + sigma * C_isqrt @ Y.T
                X = x_mean + sigma * (C_isqrt @ Y.T).T
                X = np.clip(X, -5, 5)

                # Evaluate
                Fs = []
                for i in range(n):
                    if calls_used >= remaining_budget:
                        break
                    fi = float(func(X[i]))
                    Fs.append(fi)
                    calls_used += 1
                    if fi < local_best_f:
                        local_best_f = fi
                        local_best_x = X[i].copy()
                    if fi < best_f:
                        best_f = fi
                        best_x = X[i].copy()

                if len(Fs) < n:
                    break

                Fs = np.array(Fs)
                order = np.argsort(Fs)
                Y = Y[order]
                X_sel = X[order]

                # Weighted recombination
                x_new = (w[:, None] * X_sel[:mu]).sum(axis=0)

                # Update evolution paths
                # p_sigma
                p_sigma = (1 - cs) * p_sigma + \
                    np.sqrt(cs * (2 - cs) * mueff) * (C_isqrt @ (x_new - x_mean) / sigma)
                # Hessian-norm-based sigma update
                sigma *= np.exp(cs / (sqrt_d + 1) *
                                (np.linalg.norm(p_sigma) / chi_d - 1))
                # Clamp sigma
                sigma = np.clip(sigma, 1e-8, 50)

                # p_c
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1)))) < \
                    (1 + 2 / (d + 1)) * chi_d
                p_c = (1 - cs) * p_c + hsig * np.sqrt(cs * (2 - cs) * mueff) * (x_new - x_mean) / sigma

                # Update covariance
                delta_hsig = (1 - hsig) * cs * (2 - cs)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C
                C += c1 * (np.outer(p_c, p_c) - C)
                C += cmu * (w[:, None] * (Y[:mu] * (w * (Y[:mu]).sum(axis=1) / mueff)[:, None])).sum(axis=0)

                # Symmetrize
                C = (C + C.T) / 2

                # Update C_isqrt (every few iterations for speed, or every time for d=5)
                # For d=5, SVD is cheap
                try:
                    evals, evecs = np.linalg.eigh(C)
                    evals = np.clip(evals, 1e-10, None)
                    C_isqrt = evecs @ np.diag(1 / np.sqrt(evals)) @ evecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(d)
                    C_isqrt = np.eye(d)

                # Update mean
                x_mean = x_new

                # Check for stagnation (restart trigger)
                # If best hasn't improved in many iterations, break to restart
                # We'll handle restarts in the outer loop

            return local_best_x, local_best_f

        # Multi-start CMA-ES with restarts
        # Phase 1: Quick random sampling to get a good starting point
        n_init = min(20, budget // 5)
        init_xs = rng.uniform(-5, 5, (n_init, dim))
        init_fs = np.array([float(func(x)) for x in init_xs])
        calls_used += n_init
        best_idx = np.argmin(init_fs)
        best_x = init_xs[best_idx].copy()
        best_f = init_fs[best_idx]

        # Phase 2: CMA-ES from best point
        # Try multiple CMA-ES runs with different starting points if budget allows
        # First run from global best
        remaining = budget - calls_used
        if remaining > lam:
            x0 = best_x.copy()
            sigma0 = 1.5
            _, _ = cma_run(x0, sigma0, budget)

        # If we still have budget, try a second CMA-ES from a different point
        # or from the current best with different sigma
        remaining = budget - calls_used
        if remaining > lam * 2:
            # Try from a perturbed version of best
            x0 = best_x + 0.5 * rng.standard_normal(dim)
            x0 = np.clip(x0, -5, 5)
            sigma0 = 1.0
            _, _ = cma_run(x0, sigma0, budget)

        # Final: if budget remains, do a fine local search around best
        remaining = budget - calls_used
        if remaining > 10:
            # Local refinement with small sigma
            sigma = 0.3
            x = best_x.copy()
            fx = best_f
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = float(func(y))
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
