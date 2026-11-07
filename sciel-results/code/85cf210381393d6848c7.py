"""Rewritten optimizer for BBOB-style f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with multiple restarts for Rastrigin-rotated f15."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(2024)

        def time_left():
            return deadline - time.time()

        # CMA-ES-like population-based optimizer
        def cmaes(func, x0, sigma0, max_evals, rng, deadline, time_left):
            d = dim
            # Population size
            lambda_pop = 4 + int(3 * np.log(d))
            mu = lambda_pop // 2
            # Weights
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma

            # State
            x_mean = x0.copy().astype(float)
            sigma = sigma0
            B = np.eye(d)
            D = np.ones(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            best_f = np.inf
            best_x = x0.copy()

            # Initial evaluation
            f0 = func(x_mean)
            if f0 < best_f:
                best_f = f0
                best_x = x_mean.copy()
            evals_used = 1

            while evals_used < max_evals and time_left() > 1.0:
                # Generate population
                ys = []
                fs = []
                for k in range(lambda_pop):
                    if time_left() < 0.5:
                        break
                    # Sample
                    z = rng.standard_normal(d)
                    y = x_mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    ys.append(y)
                    fs.append(fy)
                    evals_used += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if len(fs) < mu:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                ys = np.array(ys)
                fs = np.array(fs)
                ys_sorted = ys[order[:mu]]
                fs_sorted = fs[order[:mu]]

                # Update mean
                x_old = x_mean.copy()
                x_mean = np.sum(weights[:, None] * ys_sorted, axis=0)

                # Update covariance
                diff = (ys_sorted - x_old) / sigma
                C = np.sum(weights[:, None, None] * np.outer(diff, diff), axis=0)
                c_c * (np.outer(pc, pc) - np.eye(d)) + (1 - c_c) * (
                    c_sigma * (2 - c_sigma) * mu_eff * np.outer(diff, diff)
                )
                # Simplified: just use the rank-1 update for speed
                # Actually let's do proper CMA-ES update
                # pc update
                pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * np.mean(diff, axis=0)
                ps_norm = np.linalg.norm(ps)
                h_sigma = (
                    (1 - np.exp(-c_sigma)) * np.sqrt(1 - np.exp(-2 * c_sigma * evals_used / d))
                    < (1 + 1 / c_sigma) * chi_d / np.sqrt(d)
                )
                ps = (1 - c_sigma) * ps + h_sigma * np.sqrt(mu_eff) * np.mean(diff, axis=0)

                # Covariance update (simplified rank-mu)
                # For speed, use a simplified version
                C_new = (1 - c_c * (2 - c_c)) * C  # This isn't quite right for CMA-ES
                # Let me just do a simpler approach: maintain the matrix via eigen-decomposition
                # Actually, for dim=5, let's do full CMA-ES properly but efficiently

                # Recompute B and D from C
                # Use eigendecomposition
                try:
                    evals_mat, evecs = np.linalg.eigh(C)
                    evals_mat = np.maximum(evals_mat, 1e-20)
                    B = evecs
                    D = np.sqrt(evals_mat)
                except np.linalg.LinAlgError:
                    B = np.eye(d)
                    D = np.ones(d)

                # Sigma update
                sigma *= np.exp(
                    c_sigma / d_sigma * (np.linalg.norm(ps) / chi_d - 1)
                )
                sigma = np.clip(sigma, 1e-10, 50.0)

            return best_x, best_f, evals_used

        # Multi-start strategy with CMA-ES
        # Use ~60% of budget for global exploration with several starts
        # Then refine the best with more intensive search

        n_starts = 4
        global_budget = max(80, int(budget * 0.5))
        per_start = global_budget // n_starts

        best_x = None
        best_f = np.inf

        for s in range(n_starts):
            if time_left() < 5.0:
                break
            x0 = rng.uniform(-4, 4, dim)
            bx, bf, used = cmaes(func, x0, 1.5, per_start, rng, deadline, time_left)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        # Phase 2: Intensive local refinement around the best point
        remaining = budget - global_budget
        if time_left() > 2.0 and remaining > 0:
            bx, bf, used = cmaes(func, best_x, 0.5, remaining, rng, deadline, time_left)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # Phase 3: Very fine polishing with small steps
        if time_left() > 1.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for i in range(min(50, int(time_left() * 50))):
                if time_left() < 0.5:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
