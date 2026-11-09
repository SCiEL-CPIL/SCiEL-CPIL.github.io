"""Optimizer for BBOB-style Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s kill

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf

        # We'll run multiple CMA-ES restarts
        # Each restart gets a portion of the budget
        # Strategy: start with a large initial population, then adapt

        def cma_es_run(x0, sigma0, budget_here, rng):
            """Run CMA-ES for budget_here evaluations."""
            dim = self.dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(dim)))
            mu = lam // 2
            # Weights
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            cc = (4 + mueff / dim) / (dim + 4.3 + mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            x = np.array(x0, dtype=float)
            sigma = sigma0
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)

            best_x = x.copy()
            best_f = func(x)
            evals = 1

            while evals < budget_here:
                # Check time
                if time.time() > deadline:
                    break

                # Generate population
                ys = []
                for _ in range(lam):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (B @ (D * z))
                    ys.append(y)

                # Evaluate
                fs = []
                for y in ys:
                    if time.time() > deadline:
                        break
                    fy = func(np.clip(y, -5, 5))
                    fs.append(fy)
                    evals += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if len(fs) < lam:
                    break

                fs_arr = np.array(fs)
                order = np.argsort(fs_arr)
                ys_arr = np.array(ys)

                # Update mean
                yw = ys_arr[order[:mu]]
                x_new = np.sum(w[:, None] * yw, axis=0)

                # Update covariance
                ydiff = (yw - x) / sigma
                delta_x = x_new - x

                # CMA update
                pc = (1 - cc) * pc + cc / np.sqrt(cc) * delta_x / sigma
                hsig = (np.sqrt((1 - (1 - cc) ** (evals + 1)) / (1 - (1 - cc) ** (evals + 1))) /
                         (chi_n * sigma) * np.linalg.norm(ps) if sigma > 0 else 0)
                # Simplified hsig check
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * delta_x
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) / chi_n < 1.4 + 2 / (dim + 1)

                if hsig:
                    pc = (1 - cc) * pc + cc / np.sqrt(cc) * delta_x / sigma

                # Rank-mu update
                C_update = c1 * (np.outer(pc, pc) + (cc * (2 - cc) - 1) * np.eye(dim) - np.diag(D ** 2) * (B.T @ B))
                for i in range(mu):
                    vi = ydiff[i]
                    C_update += cmu * w[i] * np.outer(vi, vi)

                # Eigendecomposition
                D2, B2 = np.linalg.eigh(C_update)
                D2 = np.sqrt(np.maximum(D2, 1e-12))
                D = D2
                B = B2

                x = x_new

                # Sigma update
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)

            return best_x, best_f

        # Strategy: multiple restarts with different initializations
        # First run: broad exploration
        rng = np.random.default_rng(42)

        # Phase 1: Try several random starting points with small budgets
        # to find a good basin
        n_restarts = 5
        budget_per_restart = self.budget // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if r == 0:
                # Start at origin (Rastrigin global min is at 0)
                x0 = np.zeros(self.dim)
                sigma0 = 2.0
            else:
                x0 = rng.uniform(-4, 4, self.dim)
                sigma0 = 1.5

            bx, bf = cma_es_run(x0, sigma0, budget_per_restart, rng)
            if bf < global_best_f:
                global_best_f = bf
                global_best_x = bx.copy()

        # Phase 2: Fine-tune around the best found point with remaining budget
        remaining = self.budget - (self.budget // n_restarts) * n_restarts
        if remaining > 10 and global_best_x is not None:
            # Use a smaller sigma for fine-tuning
            rng2 = np.random.default_rng(123)
            bx, bf = cma_es_run(global_best_x, 0.5, remaining, rng2)
            if bf < global_best_f:
                global_best_f = bf
                global_best_x = bx.copy()

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
