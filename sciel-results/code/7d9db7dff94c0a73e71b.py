"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid CMA-ES + local refinement optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        calls = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls += 1
            x = np.clip(x, -self.bound, self.bound)
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # --- Phase 1: CMA-ES ---
        dim = self.dim
        # Initial population size
        lambda_ = 4 + int(3 * np.log(dim))
        # Initial step size
        sigma = 2.0
        # Mean (start at center)
        mean = np.zeros(dim)

        # CMA-ES parameters
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2 / (dim + 1.3)) / ((dim + 1) ** 2 + mu_eff + 2)
        c_mu = min(1 - c_1,
                  (2 + (2 / (dim + 1.3) - 1) * np.sqrt((mu_eff - 2) / (dim + 2))) /
                  ((dim + 2) ** 2 + mu_eff + 2))

        # Hessian update parameters
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial covariance
        C = np.eye(dim)
        # Eigen decomposition cache
        eigvals = np.ones(dim)
        eigvecs = np.eye(dim)

        # Step size control
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        d_amps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Population storage
        pop = []
        pop_f = []

        # CMA-ES loop
        max_iter = int(self.budget * 0.7 / lambda_)
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            if calls >= int(self.budget * 0.7):
                break

            # Sample population
            pop = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                # Use eigendecomposition for sampling
                y = eigvecs @ (np.sqrt(eigvals) * z)
                x = mean + sigma * y
                x = np.clip(x, -self.bound, self.bound)
                pop.append(x)

            # Evaluate
            pop_f = [eval_func(x) for x in pop]

            # Sort by fitness
            order = np.argsort(pop_f)
            pop = [pop[i] for i in order]
            pop_f = [pop_f[i] for i in order]

            # Update mean
            mean_old = mean.copy()
            for i in range(mu):
                mean += weights[i] * pop[i]
            mean = np.clip(mean, -self.bound, self.bound)

            # Update step size
            y_mean = (mean - mean_old) / sigma if sigma > 0 else np.zeros(dim)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            sigma *= np.exp(c_sigma / d_amps * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-10)

            # Update path
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - mean_old) / sigma
            y_c = p_c / np.sqrt(np.sum(p_c ** 2) + 1e-30)

            # Update covariance
            delta = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1)) * (c_c - 1)
            C = ((1 - c_1 - c_mu + delta) * C +
                 c_1 * (np.outer(p_c, p_c) - C) +
                 c_mu / sigma ** 2 * sum(weights[i] * np.outer(pop[i] - mean_old, pop[i] - mean_old) for i in range(mu)))

            # Periodic eigen decomposition
            if iteration % 5 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)

        # --- Phase 2: Local refinement with adaptive step size ---
        # Use remaining budget for local search
        remaining_budget = self.budget - calls
        if remaining_budget <= 0:
            return best_x, best_f

        # Start from best point found
        x = best_x.copy()
        f = best_f

        # Adaptive local search with multiple step sizes
        step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        current_step_idx = 0

        while calls < self.budget and time.time() < deadline:
            if current_step_idx >= len(step_sizes):
                break

            step = step_sizes[current_step_idx]
            improvement_found = False

            # Try random directions
            for _ in range(20):
                if calls >= self.budget or time.time() > deadline:
                    break

                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction) + 1e-30

                # Try moving in this direction
                x_new = x + step * direction
                x_new = np.clip(x_new, -self.bound, self.bound)
                f_new = eval_func(x_new)

                if f_new < f:
                    x = x_new
                    f = f_new
                    best_x = x.copy()
                    best_f = f
                    improvement_found = True

                # Also try opposite direction
                x_new2 = x - step * direction
                x_new2 = np.clip(x_new2, -self.bound, self.bound)
                f_new2 = eval_func(x_new2)

                if f_new2 < f:
                    x = x_new2
                    f = f_new2
                    best_x = x.copy()
                    best_f = f
                    improvement_found = True

            if not improvement_found:
                # Reduce step size
                current_step_idx += 1
            else:
                # Keep same step size if improvement found, but limit iterations
                if calls >= self.budget or time.time() > deadline:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
