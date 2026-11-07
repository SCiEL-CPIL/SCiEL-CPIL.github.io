"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with restarts for f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(12345)
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # We'll use multiple restarts of a simple CMA-ES-like procedure
        # Each restart gets a fraction of the budget
        num_restarts = 5
        budget_per_restart = budget // num_restarts

        # Initial global exploration: evaluate a few points to find a good starting region
        # Use a small set of random points to identify promising areas
        init_points = []
        for _ in range(min(10, budget_per_restart // 2)):
            x = rng.uniform(-4.5, 4.5, n)
            fx = func(x)
            init_points.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort by fitness, take the best few as starting points for CMA-ES
        init_points.sort(key=lambda t: t[0])

        for restart_idx in range(num_restarts):
            # Check time
            if time.time() > deadline:
                break

            # Pick a starting point: mix of best found and random
            if restart_idx == 0:
                x_mean = best_x.copy() if best_x is not None else rng.uniform(-4.5, 4.5, n)
            elif restart_idx < len(init_points):
                x_mean = init_points[restart_idx % len(init_points)][1].copy()
            else:
                x_mean = rng.uniform(-4.5, 4.5, n)

            fx_mean = func(x_mean)
            if fx_mean < best_f:
                best_f = fx_mean
                best_x = x_mean.copy()

            # CMA-ES parameters
            lam = 12  # population size
            sigma = 1.0
            c1 = (4 + np.log(n)) / (n + 4) / (n + 4) / 2
            cmu = min(0.2, (n - 2 + 2.0 / (n + 1.0)) / (n + 4) / (n + 4) / 2)
            cs = (n + 1.5) / (n + 9)
            c1 += (1 - c1) * min(1, (lam / 2 - 1) / (n + n - 2))
            cmu = min(0.2, (n - 2 + 2.0 / (n + 1.0)) / (n + 4) / (n + 4) / 2)
            damps = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (n + 1) - 1) - 1) + cs

            C = np.eye(n)
            C_sqrt_inv = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)

            # Track calls used in this restart
            calls_in_restart = 0
            max_calls_restart = budget_per_restart - 2  # reserve some for next restarts

            while calls_in_restart < max_calls_restart:
                if time.time() > deadline:
                    break

                # Generate population
                ys = []
                fvals = []
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (C_sqrt_inv @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    ys.append(y)
                    fvals.append(fy)
                    calls_in_restart += 1

                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if len(ys) < 2:
                    break

                # Sort by fitness
                idx = np.argsort(fvals)
                ys_sorted = [ys[i] for i in idx]
                fvals_sorted = [fvals[i] for i in idx]
                mu = lam // 2  # number of recombination weights

                # Recombination weights (logarithmic)
                weights = np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mueff = 1.0 / np.sum(weights**2)

                # Update mean
                x_new = sum(w * y for w, y in zip(weights, ys_sorted[:mu]))
                x_new = np.clip(x_new, -5, 5)

                # Update evolution path
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt_inv.T @ ((x_new - x_mean) / sigma))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / np.sqrt(1 - (1 - cs)**(2 * (calls_in_restart // lam + 1)))) < (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - cs)**2)
                pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * (x_new - x_mean) / sigma

                # Update covariance matrix
                delta_d = (2 + 2.0 / (n + 1)) if hsig else 0
                rank_mu = np.outer(x_new - x_mean, x_new - x_mean)
                rank_pc = np.outer(pc, pc)
                C = (1 + c1 * delta_d + cmu - c1 - cmu) * C + c1 * (rank_pc - (1 - delta_d) * C) + cmu * rank_mu

                # Ensure positive semi-definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update square root inverse
                eigvals, eigvecs = np.linalg.eigh(C)
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                x_mean = x_new

                # Stagnation check: if no improvement in last lam*3 evaluations, break
                # Simple heuristic: if sigma is very small or very large, break
                if sigma < 1e-6 or sigma > 50:
                    break

                # If the mean hasn't changed much, we might be stuck
                # Check if best_f has improved in last few generations
                # For simplicity, just run until budget or time

        return best_x, best_f
# EVOLVE-BLOCK-END
