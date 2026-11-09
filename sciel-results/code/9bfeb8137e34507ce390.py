"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population, adaptation, and time-safe budgeting."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        # Population size
        lambda_ = int(4 + 3 * np.log(dim))
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        c_1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) / 10
        c_mu = min(1 - c_1, 2 * (mu_eff - 2) / ((dim + 2) ** 2 + mu_eff))
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        h_sigma = 1.4 + 2 / (dim + 4)

        # Initial parameters
        x0 = np.zeros(dim)
        sigma0 = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))

        best_x = x0.copy()
        best_f = np.inf
        rng = np.random.default_rng(42)

        total_budget = budget
        remaining = total_budget

        # We will run CMA-ES until budget or time is exhausted.
        # We might restart if it stagnates, but CMA-ES is robust.
        # For dim=5, one run is usually sufficient if initialized well.
        # However, to be safe against local optima or bad init, we can do a few restarts.
        # Given the budget of 500, we can afford ~5-10 runs of lambda=10.
        # Let's try 3 restarts if stagnation is detected.

        max_restarts = 3
        restarts_done = 0

        while remaining > 0 and restarts_done < max_restarts and time.time() < deadline:
            # Reset CMA-ES state for this restart
            if restarts_done == 0:
                x = np.zeros(dim)
            else:
                # Start near previous best but with some noise
                x = best_x + rng.normal(0, 0.5, dim)
                x = np.clip(x, -5, 5)

            fx = func(x)
            remaining -= 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            sigma = sigma0

            # Main CMA-ES loop
            while remaining > 0 and time.time() < deadline:
                # Check for time
                if time.time() > deadline - 2.0:
                    break

                # Generate population
                offspring = []
                offspring_f = []

                for j in range(lambda_):
                    if remaining <= 0:
                        break
                    z = rng.multivariate_normal(np.zeros(dim), C)
                    y = x + sigma * z
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    remaining -= 1
                    offspring.append(y)
                    offspring_f.append(fy)

                if len(offspring) == 0:
                    break

                offspring_f = np.array(offspring_f)
                # Sort
                sorted_idx = np.argsort(offspring_f)
                sorted_offspring = [offspring[i] for i in sorted_idx]
                sorted_f = offspring_f[sorted_idx]

                # Update mean
                y_w = np.sum(w[:mu] * np.array(sorted_offspring[:mu]), axis=0)
                x_new = y_w

                # Update covariance matrix
                diff = np.array(sorted_offspring[:mu]) - x
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_sigma) * h_sigma * chi_n ** 2 * C) \
                    + c_mu * np.sum(w[:mu] * np.array([np.outer(diff[i], diff[i]) for i in range(mu)]), axis=0)

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                # Add small regularization if needed
                C += 1e-10 * np.eye(dim)

                # Update step size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (np.linalg.inv(np.sqrt(C)) @ (x_new - x) / sigma)
                ps_norm = np.linalg.norm(ps)
                sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))

                # Update evolution path
                pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (np.linalg.inv(np.sqrt(C)) @ (x_new - x) / sigma)

                # Update x
                x = x_new
                fx = sorted_f[0]

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Stagnation check: if sigma is very small or no improvement in last 20 iterations
                # Simple check: if sigma < 1e-8, break
                if sigma < 1e-8:
                    break

                # Check for progress
                # If we haven't improved best_f in last 2*lambda_ iterations, consider restarting
                # We don't track history here easily, so we rely on sigma convergence or time.
                # For now, if sigma drops below 1e-4, we assume convergence for this restart.
                if sigma < 1e-4 and remaining > lambda_:
                    # Try a small perturbation to escape potential local min?
                    # Or just break and restart if budget allows.
                    break

            restarts_done += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
