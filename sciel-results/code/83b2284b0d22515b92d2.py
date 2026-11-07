"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Rank-One CMA-ES global search + Nelder-Mead/(1+1)-ES local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        max_time = 295.0

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        def check_time():
            return time.time() - start_time >= max_time or calls >= budget

        def eval_func(x):
            nonlocal calls, best_f, best_x
            if check_time():
                return float('inf')
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # ---- Phase 1: Coarse global sampling to find promising basins ----
        # Lunacek bi-Rastrigin has two global minima; sample broadly.
        n_coarse = min(80, budget // 6)
        candidates = []

        for i in range(n_coarse):
            if check_time():
                break
            if i % 3 == 0:
                x = rng.uniform(-5, 5, dim)
            elif i % 3 == 1:
                center = rng.choice([-3.5, 3.5, 0.0])
                x = np.full(dim, center) + rng.normal(0, 1.5, dim)
            else:
                x = rng.uniform(-4.5, 4.5, dim)
            x = np.clip(x, -5, 5)
            fx = eval_func(x)
            if np.isfinite(fx):
                candidates.append((fx, x.copy()))

        if not candidates:
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])

        # ---- Phase 2: Rank-One CMA-ES from the best coarse seed ----
        # Adaptive covariance via rank-one update; more efficient than
        # population-based ES for low-dim multimodal problems.

        mu = 4
        lam = 16
        sigma = 1.5
        sigma_min = 1e-10
        sigma_max = 5.0

        # CMA-ES parameters
        c1 = (2 / ((dim + 1) ** 2 + 5)) * 0.4
        cmu = min(0.4 - c1, 2 / ((dim + 2) ** 2 + 13)) * 0.4
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        mu_eff = mu / 2.0

        # Initialize mean from best coarse candidate
        mean = candidates[0][1].copy()
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        step_size = np.ones(dim)

        # Build initial population around best
        population = [(candidates[0][0], mean.copy())]
        for i in range(1, min(mu, len(candidates))):
            population.append((candidates[i][0], candidates[i][1].copy()))
        while len(population) < mu and not check_time():
            x = mean + sigma * rng.standard_normal(dim)
            x = np.clip(x, -5, 5)
            fx = eval_func(x)
            if np.isfinite(fx):
                population.append((fx, x.copy()))
        population.sort(key=lambda t: t[0])

        # Weighted recombination weights
        weights = np.log(mu / 2 + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Evolution path for sigma adaptation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        global_best_f = best_f
        stagnation = 0
        max_stagnation = 40

        while not check_time():
            # Generate offspring
            offspring = []
            for j in range(lam):
                if check_time():
                    break
                z = rng.standard_normal(dim)
                # Transform using C^{-1/2} (approximate via step_size scaling)
                y = mean + sigma * (step_size * z)
                y = np.clip(y, -5, 5)
                fy = eval_func(y)
                if np.isfinite(fy):
                    offspring.append((fy, y.copy(), z.copy()))

            if not offspring:
                break

            # Sort offspring
            offspring.sort(key=lambda t: t[0])

            # Select best mu
            selected = offspring[:mu]

            # Compute weighted mean of selected
            y_w = np.zeros(dim)
            for w, (_, y, _) in zip(weights, selected):
                y_w += w * (y - mean)
            new_mean = mean + y_w

            # Update evolution path for sigma
            sigma_old = sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (
                C_inv_sqrt @ y_w) / sigma
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (calls - n_coarse))) < (1 + 1.4 / dim) * np.sqrt(dim + 2) - 1 else 0

            # Update sigma
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
            sigma = np.clip(sigma, sigma_min, sigma_max)

            # Update covariance (rank-one + rank-mu)
            y_selected = np.array([y for _, y, _ in selected])
            C_old = C.copy()
            C = (1 - c1 - cmu) * C + c1 * np.outer(y_w, y_w) / sigma ** 2
            for w, y in zip(weights, y_selected):
                C += cmu * w * np.outer((y - mean) / sigma, (y - mean) / sigma)

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update C_inv_sqrt approximation
            eigvals2, eigvecs2 = np.linalg.eigh(C)
            eigvals2 = np.maximum(eigvals2, 1e-20)
            C_inv_sqrt = eigvecs2 @ np.diag(1.0 / np.sqrt(eigvals2)) @ eigvecs2.T
            step_size = np.sqrt(np.diag(C_inv_sqrt @ C_inv_sqrt))

            # Update mean
            mean = new_mean
            mean = np.clip(mean, -5, 5)

            # Track improvement
            if selected[0][0] < global_best_f - 1e-12:
                global_best_f = selected[0][0]
                stagnation = 0
            else:
                stagnation += 1

            # Stagnation restart
            if stagnation >= max_stagnation:
                mean = best_x.copy()
                sigma = 1.5
                C = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                step_size = np.ones(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stagnation = 0
                max_stagnation = 25

        # ---- Phase 3: Local polish ----
        remaining_budget = budget - calls
        if remaining_budget > 10 and not check_time():
            # Nelder-Mead local optimization
            try:
                result = minimize(
                    lambda x: eval_func(x),
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining_budget // 2, 300),
                             'fatol': 1e-12,
                             'xatol': 1e-10,
                             'adaptive': True}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

            # (1+1)-ES fine-tuning
            if not check_time():
                sigma_fine = 0.05
                x = best_x.copy()
                fx = best_f
                no_improve = 0
                max_no_improve = 40

                while not check_time():
                    direction = rng.standard_normal(dim)
                    candidate = x + sigma_fine * direction
                    candidate = np.clip(candidate, -5, 5)
                    fc = eval_func(candidate)
                    if np.isfinite(fc) and fc < fx:
                        x = candidate
                        fx = fc
                        no_improve = 0
                        sigma_fine = min(sigma_fine * 1.02, 0.3)
                    else:
                        no_improve += 1
                        sigma_fine = max(sigma_fine * 0.93, 1e-7)
                    if no_improve >= max_no_improve:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
