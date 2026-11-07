"""CMA-ES with time guard and restarts for BBOB f6."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with adaptive sigma, time guard, and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin below 330 s

        # Population size per CMA-ES convention
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2

        # Cumulative weight coefficients
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def run_cmaes(x0, sigma0, eval_budget):
            """Run CMA-ES for at most eval_budget function evaluations."""
            x = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            best_x = x0.copy()
            best_f = np.inf
            calls = 0

            for gen in range(eval_budget // lam + 2):
                if calls >= eval_budget:
                    break
                if time.time() > deadline:
                    break

                # Sample population
                offspring = []
                for i in range(lam):
                    z = np.random.randn(dim)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    offspring.append((y, i))

                # Evaluate
                fvals = []
                for y, i in offspring:
                    if calls >= eval_budget:
                        break
                    if time.time() > deadline:
                        break
                    fy = func(y)
                    fvals.append((fy, i, y))
                    calls += 1

                if len(fvals) == 0:
                    break

                # Sort
                fvals.sort(key=lambda t: t[0])

                # Update best
                if fvals[0][0] < best_f:
                    best_f = fvals[0][0]
                    best_x = fvals[0][2].copy()

                # Take top mu
                top = fvals[:mu]
                y_w = np.array([t[2] for t in top])
                f_w = np.array([t[0] for t in top])

                # New mean
                x_new = np.sum(weights[:, None] * y_w, axis=0)

                # Standard deviation of selected points
                d = np.sqrt((x_new - x) @ np.linalg.inv(np.eye(dim)) @ (x_new - x)) if dim <= 1 else np.linalg.norm(x_new - x)
                # Simpler: use the norm
                d = np.linalg.norm(x_new - x)

                # Update ps
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (B @ (D * ((x_new - x) / sigma)))

                # Update pc
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) / chi_n < (1 + 2 / (dim + 1)) * c_sigma
                pc = (1 - c_c) * pc + hsig * c_c * (x_new - x) / sigma

                # Update B, D via rank-mu + rank-one update
                C = (B @ np.diag(D ** 2) @ B.T)  # current C
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * np.eye(dim))
                C += c_mu * np.sum(weights[:, None] * np.outer(y_w - x, y_w - x), axis=0) / sigma ** 2

                # Symmetrize
                C = (C + C.T) / 2.0

                # Eigen-decomposition
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-18)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                except np.linalg.LinAlgError:
                    break

                x = x_new
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 50.0)

            return best_x, best_f, calls

        # Initial population: a few diverse starting points
        rng = np.random.default_rng(42)
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        # Phase 1: broad exploration with multiple restarts
        n_restarts = 4
        evals_per_restart = max(10, budget // n_restarts)

        for r in range(n_restarts):
            if total_calls >= budget:
                break
            if time.time() > deadline:
                break

            # Start point: random in box, but bias toward center after first restart
            if r == 0:
                x0 = rng.uniform(-5, 5, dim)
            else:
                # Start near current best with some perturbation
                x0 = np.clip(best_x + rng.uniform(-2, 2, dim), -5, 5)

            sigma0 = 2.0 if r == 0 else max(0.5, 2.0 * (0.5 ** r))
            remaining = budget - total_calls
            bx, bf, used = run_cmaes(x0, sigma0, remaining)
            total_calls += used

            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
