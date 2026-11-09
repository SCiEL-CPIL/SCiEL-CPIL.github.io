"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        rng = np.random.default_rng(12345)

        # Initial population: center + a few random points to estimate scale
        x0 = np.zeros(dim)
        f0 = func(x0)

        # Estimate initial sigma from a few random evaluations
        samples = []
        for _ in range(10):
            s = rng.uniform(-5, 5, dim)
            fs = func(s)
            samples.append((s, fs))
        # Use spread of points with similar fitness as initial step size
        sorted_samples = sorted(samples, key=lambda t: t[1])
        good = [s for s, f in sorted_samples[:5]]
        if len(good) >= 2:
            diffs = [good[i] - good[0] for i in range(1, len(good))]
            sigma = max(0.5, np.mean([np.linalg.norm(d) / np.sqrt(dim) for d in diffs]))
        else:
            sigma = 2.0

        # Start CMA-ES
        x = x0.copy()
        fx = f0
        best_x = x0.copy()
        best_f = f0

        # CMA-ES parameters
        lambda_ = 4 * (dim + 2)  # population size
        mu = lambda_ // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / ((dim + 1.3) ** 2 + mueff)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.7) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        evals = 0
        total_calls = 11  # 1 for x0 + 10 for sigma estimation

        while total_calls < budget and time.time() < deadline:
            # Generate population
            B = rng.standard_normal((lambda_, dim))
            D = np.linalg.cholesky(C)
            pop = [x + sigma * (B[i] @ D) for i in range(lambda_)]
            # Clip to bounds
            pop = [np.clip(p, -5, 5) for p in pop]

            # Evaluate
            pop_f = []
            for p in pop:
                if total_calls >= budget or time.time() >= deadline:
                    break
                fp = func(p)
                pop_f.append(fp)
                total_calls += 1
                if fp < best_f:
                    best_f = fp
                    best_x = p.copy()

            if len(pop_f) < 2:
                break

            # Sort by fitness
            idx = np.argsort(pop_f)
            selected = [pop[i] for i in idx[:mu]]
            selected_f = [pop_f[i] for i in idx[:mu]]

            # Mean of selected
            m_new = sum(w * selected[i] for i, w in enumerate(weights))

            # Update evolution path ps
            D_inv = np.linalg.inv(D)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (D_inv @ (m_new - x) / sigma)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(1e-10, min(sigma, 100))

            # Update evolution path pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (m_new - x) / sigma

            # Update covariance matrix
            delta = (1 + 2 * (c_c == 0) * (np.linalg.norm(pc) ** 2 / dim - 1 > 14.3 / dim)) * c_c
            rank1 = np.outer(pc, pc)
            rankmu = sum(weights[i] * np.outer(selected[i] - x, selected[i] - x) for i in range(mu))
            C = (1 + c_1 + c_mu - c_1) * C + c_1 * (rank1 - C) + c_mu * rankmu

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x = m_new
            fx = min(selected_f)

        return best_x, best_f
# EVOLVE-BLOCK-END
