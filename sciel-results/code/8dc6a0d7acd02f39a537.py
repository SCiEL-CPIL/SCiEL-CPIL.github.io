"""Optimizer for BBOB-style f17 (Schaffers F7) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 330s limit
        d = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        # CMA-ES parameters tuned for dim=5
        lam = 10
        mu = lam // 2

        cc = (4 + d / (d + 4)) / ((d + 1.3) + 2 / (d + mu / 2))
        cs = (mu / 2 + 1) / (d + mu / 2 + 2)
        c1 = 2 / ((d + 1.3) ** 2 + mu)
        cmu = min(1 - c1, 2 * (mu / 2 - 1) / ((d + 2) ** 2 + mu))
        damps = 1 + 2 * max(0, np.sqrt((mu / 2 - 1) / (d + 1)) - 1) + cs
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Precompute selection weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1 / np.sum(w ** 2)

        def cmaes_run(x0, sigma0, budget_for_this_run, rng):
            nonlocal best_x, best_f, calls_made

            x = x0.copy()
            sigma = sigma0
            C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)

            f_best = np.inf
            x_best = x.copy()
            no_improve = 0
            max_no_improve = 15  # generations without improvement before stopping

            for gen in range(budget_for_this_run // lam + 1):
                if time.time() > deadline:
                    break
                if calls_made >= total_budget:
                    break

                # Generate lambda offspring using eigendecomposition for stability
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                sqrt_eigvals = np.sqrt(eigvals)
                B = (eigvecs * sqrt_eigvals) @ eigvecs.T

                offspring = []
                for i in range(lam):
                    zi = rng.standard_normal(d)
                    yi = x + sigma * (B @ zi)
                    yi = np.clip(yi, -5.0, 5.0)
                    fi = func(yi)
                    calls_made += 1
                    offspring.append((fi, yi))

                # Sort offspring
                offspring.sort(key=lambda t: t[0])

                # Update best
                if offspring[0][0] < f_best:
                    f_best = offspring[0][0]
                    x_best = offspring[0][1].copy()
                    no_improve = 0
                else:
                    no_improve += 1
                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()

                # Stagnation detection
                if no_improve > max_no_improve:
                    break

                # Selection
                x_new = np.zeros(d)
                for i in range(mu):
                    x_new += w[i] * offspring[i][1]

                # Check for stagnation
                if np.linalg.norm(x_new - x) < 1e-14:
                    break

                # Update evolution paths
                if sigma > 1e-14:
                    diff = (x_new - x) / sigma
                    z_mean = (eigvecs * (1.0 / np.maximum(sqrt_eigvals, 1e-12))) @ (eigvecs.T @ diff)
                else:
                    z_mean = np.zeros(d)

                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * z_mean
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * z_mean

                # Update covariance matrix
                ps_norm = np.linalg.norm(ps)
                hc_thresh = (1.4 + 2 / (d + 1)) * chiN
                delta_hc = (1 - cc) if ps_norm / np.sqrt(max(1 - (1 - cs) ** 2, 1e-12)) < hc_thresh else 0

                C = (1 + c1 * delta_hc * cc * (2 - cc) - c1) * C + c1 * np.outer(pc, pc)
                rank_mu = np.zeros((d, d))
                for i in range(mu):
                    diff_i = offspring[i][1] - x
                    rank_mu += w[i] ** 2 * np.outer(diff_i, diff_i)
                C = C + cmu * rank_mu

                # Symmetrize and ensure PD
                C = (C + C.T) / 2
                eigvals2, eigvecs2 = np.linalg.eigh(C)
                if np.min(eigvals2) < 1e-14:
                    eigvals2 = np.maximum(eigvals2, 1e-14)
                    C = (eigvecs2 * eigvals2) @ eigvecs2.T

                # Update step size
                sigma *= np.exp(cs / damps * (ps_norm / chiN - 1))

                x = x_new

                # Convergence check
                if sigma < 1e-10:
                    break

            return x_best, f_best

        rng = np.random.default_rng(42)

        # Multi-restart strategy: 5 restarts with decreasing budget
        # First run gets more budget for global exploration
        num_restarts = 5
        budgets = []
        remaining = total_budget
        for i in range(num_restarts):
            if i == num_restarts - 1:
                b = remaining
            else:
                frac = 0.35 if i == 0 else (0.20 if i == 1 else (0.15 if i == 2 else (0.12 if i == 3 else 0.08)))
                b = int(remaining * frac)
            budgets.append(b)
            remaining -= b
        budgets[-1] = remaining

        # First run from center with large sigma
        x0 = np.zeros(d)
        sigma0 = 2.5
        cmaes_run(x0, sigma0, budgets[0], rng)

        # Restart runs: use best point + perturbation for first restart, then random
        for r in range(1, num_restarts):
            if calls_made >= total_budget or time.time() > deadline:
                break
            if r == 1 and best_x is not None:
                # Perturb best point
                x0 = best_x + 1.5 * rng.standard_normal(d)
                x0 = np.clip(x0, -4.5, 4.5)
            else:
                x0 = rng.uniform(-4.0, 4.0, d)
            sigma0 = 1.2 if r == 1 else 0.8
            cmaes_run(x0, sigma0, budgets[r], rng)

        # Final polish: adaptive (1+1)-ES around best point
        if best_x is not None and calls_made < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma_local = 0.3
            min_sigma = 1e-6
            max_rejections = 30

            rejections = 0
            while calls_made < total_budget and time.time() < deadline:
                if sigma_local < min_sigma:
                    break
                if rejections > max_rejections:
                    break

                y = np.clip(x + sigma_local * rng.standard_normal(d), -5.0, 5.0)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x = y
                    fx = fy
                    rejections = 0
                    sigma_local *= 1.15
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    rejections += 1
                    sigma_local *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
