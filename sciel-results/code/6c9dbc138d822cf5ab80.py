"""CMA-ES optimizer for BBOB f12 (Bent Cigar) with time guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES based optimizer with wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        deadline = t0 + 275.0
        max_calls = budget

        calls = [0]
        best_x = np.zeros(dim)
        best_fx = 1e18

        def f(x):
            nonlocal best_x, best_fx
            calls[0] += 1
            val = float(func(x))
            if not np.isfinite(val):
                return 1e18
            if val < best_fx:
                best_fx = val
                best_x = x.copy()
            return val

        def remaining():
            return deadline - time.time()

        def cma_es(x0, fx0, max_calls, max_iter=200):
            """Run CMA-ES from x0 with initial sigma. Returns best (x, fx)."""
            n = dim
            # Initial sigma: start moderate
            sigma = 2.0

            # Population size
            lambda_ = max(4, int(4 + 3 * np.log(n)))
            mu = lambda_ // 2  # number of parents

            # Weights
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = (2 - mu * w[0] + cc) / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            # State
            m = x0.copy().astype(float)
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Initial simplex for CMA-ES: just start with identity covariance
            # Generate initial population
            def sample_population():
                """Generate lambda samples from N(m, sigma^2 * C)."""
                # Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback: eigenvalue decomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    L = eigvecs @ np.diag(np.sqrt(eigvals))
                samples = []
                fvals = []
                for _ in range(lambda_):
                    z = np.random.randn(n)
                    x = m + sigma * (L @ z)
                    x = np.clip(x, -5, 5)
                    fx = f(x)
                    samples.append(x)
                    fvals.append(fx)
                return samples, fvals

            local_best_x = m.copy()
            local_best_fx = fx0

            for it in range(max_iter):
                if calls[0] >= max_calls or remaining() < 5.0:
                    break

                # Sample and evaluate
                pops, fvals = sample_population()

                # Sort by fitness
                idx = np.argsort(fvals)
                pops = [pops[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                # Update best
                if fvals[0] < local_best_fx:
                    local_best_fx = fvals[0]
                    local_best_x = pops[0].copy()

                # Check convergence
                if sigma < 1e-8:
                    break

                # Weighted mean
                m_new = sum(w[i] * pops[i] for i in range(mu))

                # Update evolution paths
                pc = (1 - cc) * pc + cc / np.sqrt(cc) * np.linalg.inv(np.linalg.cholesky(C) + 1e-10) @ (m_new - m)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / np.linalg.cholesky(C + 1e-10) @ (m_new - m)
                hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (it + 1)) < (1.4 + 2 / (n + 1)) * chi_n else 0.0

                # Update mean
                m = m_new

                # Update covariance
                delta_var = 1 + 2 * max(0, hsig - 1) * cc * (1 - cc)
                rank_one = np.outer(pc, pc)
                rank_mu = sum(w[i] * np.outer(pops[i] - m, pops[i] - m) for i in range(mu))
                C = (1 + c1 * delta_var - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Adapt sigma
                sigma *= np.exp(cs / (5 * n) * (np.linalg.norm(ps) / chi_n - 1))

                # Clamp sigma
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)

                # Keep mean in bounds
                m = np.clip(m, -5, 5)

            return local_best_x, local_best_fx

        # Multi-start with CMA-ES
        rng = np.random.default_rng(42)
        starts = [
            np.zeros(dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            np.array([3.0, -3.0, 2.0, -2.0, 1.0]),
        ]

        per_start = max(30, budget // len(starts))

        for si, s in enumerate(starts):
            if calls[0] >= max_calls or remaining() < 10.0:
                break
            s = np.clip(s, -5, 5)
            fx = f(s)
            x, fx = cma_es(s, fx, max_calls, max_iter=100)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # If we have budget left, do a final CMA-ES from the best point
        if remaining() > 15.0 and calls[0] < max_calls:
            x, fx = cma_es(best_x, best_fx, max_calls, max_iter=50)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
