import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or not time_ok():
                return np.inf
            calls += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Random sampling to find a good starting point
        n_random = 30
        for i in range(n_random):
            x = rng.uniform(-4.0, 4.0, size=dim)
            eval_func(x)

        # Phase 2: Multi-start CMA-ES (lightweight implementation)
        # CMA-ES is more robust than NM for multimodal landscapes
        n_restarts = 5
        cma_budget_per_run = max(10, (budget - calls - 1) // n_restarts)

        for r in range(n_restarts):
            if calls >= budget - 1 or not time_ok():
                break

            # Start CMA-ES from the current best point with small perturbation
            x0 = best_x + rng.normal(0, 0.5, size=dim)
            x0 = np.clip(x0, -5.0, 5.0)

            # CMA-ES parameters
            sigma = 1.0
            lam = max(4, int(4 + 3 * np.log(dim)))  # population size
            c1 = (2 / ((dim + 1) ** 2 + 10)) * 0.5
            cmu = min(0.3, (2 * (2 - 1 / (dim + 1)) / ((dim + 2) * (dim + 2) + 10)) * 0.5)
            cs = (lam + 2) / (dim + lam + 5)
            ds = 1 + 2 * max(0, np.sqrt((lam - 1) / (dim + 1) - 1) + cs)
            cc = (4 + dim / lam) / (dim + 4 + 2 * lam / dim)
            chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # State variables
            mean = x0.copy()
            C = np.eye(dim)
            C_sqrt_inv = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            sigma = 1.0

            # Precompute for efficiency
            def compute_C_sqrt_inv(C):
                # Compute inverse square root of C
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                return C_sqrt_inv

            C_sqrt_inv = compute_C_sqrt_inv(C)

            max_iters = cma_budget_per_run // lam

            for gen in range(max_iters):
                if calls >= budget - 1 or not time_ok():
                    break

                # Sample lambda points
                samples = np.zeros((lam, dim))
                fvals = np.zeros(lam)

                for i in range(lam):
                    z = rng.standard_normal(dim)
                    x = mean + sigma * C_sqrt_inv @ z
                    x = np.clip(x, -5.0, 5.0)
                    samples[i] = x
                    fvals[i] = eval_func(x)
                    if fvals[i] == np.inf:
                        break

                if fvals[-1] == np.inf:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                samples = samples[order]
                fvals = fvals[order]

                # Weights
                mu = lam // 2
                weights = np.zeros(lam)
                top_mu_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights[:mu] = top_mu_weights / top_mu_weights.sum()
                mueff = 1 / np.sum(weights[:mu] ** 2)

                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * samples[:mu], axis=0)

                # Update step-size control
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * C_sqrt_inv @ ((new_mean - mean) / sigma)
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (gen + 1)) < (1.4 + 2 / (dim + 1)) * chi)
                sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi - 1))

                # Update covariance matrix
                p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mueff) * (new_mean - mean) / sigma
                rank_one = np.outer(p_c, p_c)
                hsig = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - cc) ** (gen + 1)) < (1.4 + 2 / (dim + 1)) * chi)
                delta_hsig = (1 - hsig) * cc * (2 - cc)

                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = C_sqrt_inv @ (samples[i] - mean) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

                # Ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-10:
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T

                C_sqrt_inv = compute_C_sqrt_inv(C)
                mean = new_mean

        return best_x, best_f
