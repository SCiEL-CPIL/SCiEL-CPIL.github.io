"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with restarts for BBOB f12 (bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track global best
        best_x = None
        best_f = np.inf

        # CMA-ES parameters
        # For bent cigar, we want a more aggressive initial sigma
        # and faster adaptation

        total_calls = 0

        # We'll use a population-based CMA-ES
        # lambda = 4 + floor(3 * log(dim))
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2  # number of selected solutions

        # Weights
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def run_cmaes(x0, sigma0, max_calls):
            """Run CMA-ES from x0 with initial sigma0, using at most max_calls evaluations."""
            nonlocal total_calls, best_x, best_f

            x = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            cs = c_sigma
            cc = c_c
            c1 = c_1
            cmu = c_mu
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            dampc = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_c

            calls_in_run = 0
            pop_best_f = np.inf

            while calls_in_run < max_calls:
                if total_calls + lam > budget:
                    lam = budget - total_calls
                    if lam < 2:
                        break
                if time.time() > deadline:
                    break

                # Generate population
                ys = []
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    y = x + sigma * C @ rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    ys.append(y)
                    total_calls += 1
                    calls_in_run += 1

                if len(ys) < 2:
                    break

                # Evaluate
                fs = np.array([func(y) for y in ys])

                # Update global best
                min_idx = np.argmin(fs)
                if fs[min_idx] < best_f:
                    best_f = fs[min_idx]
                    best_x = ys[min_idx].copy()
                if fs[min_idx] < pop_best_f:
                    pop_best_f = fs[min_idx]

                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = np.array([ys[i] for i in order])
                fs_sorted = fs[order]

                # Update mean
                w = weights[:min(mu, lam)]
                x_new = np.sum(w[:, None] * ys_sorted[:len(w)], axis=0)

                # Update step size control vector
                pc = (1 - cs) * pc + cs / np.sqrt(1 - (1 - cs) ** 2) * (x_new - x) / sigma

                # Update covariance matrix
                ps = (1 - c_sigma) * ps + np.sqrt(cs * (2 - cs) * mueff) / np.linalg.norm(pc + 1e-15) * (x_new - x) / sigma

                # Check if step size should be adapted
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 1 / dim) * chi_n:
                    sigma *= np.exp(c_sigma / damps * (np.linalg.norm(ps) / chi_n - 1))

                # Update C
                delta_diag = 0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 1 / dim) * chi_n else 1
                C = ((1 - c1 - cmu) * C +
                     c1 * ((1 + delta_diag - 1) * np.outer(pc, pc) + pc @ pc) +
                     cmu * np.sum(w[:, None] * np.outer(ys_sorted[:len(w)] - x, ys_sorted[:len(w)] - x), axis=0))

                # Symmetrize
                C = (C + C.T) / 2

                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                x = x_new
                if sigma < 1e-15:
                    break

                # Early stopping if no improvement for many iterations
                # (simple heuristic)

            return x, sigma, pop_best_f

        # Initial phase: try a few different starting points
        # First, a random start
        x0 = rng.uniform(-5, 5, dim)
        f0 = func(x0)
        total_calls += 1
        best_f = f0
        best_x = x0.copy()

        # Run CMA-ES from this point
        remaining = budget - total_calls
        if remaining > 10:
            run_cmaes(x0, 2.0, remaining)

        return best_x, best_f
# EVOLVE-BLOCK-END
