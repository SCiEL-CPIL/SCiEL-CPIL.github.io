"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best result
        best_x = None
        best_f = np.inf

        # Total calls made
        calls_made = 0

        # CMA-ES parameters
        # lambda = 4*dim + 4
        lam = 4 * dim + 4
        # Mu = lam // 2
        mu = lam // 2
        # Weights
        rho = np.sqrt(mu)
        c_sigma = (mu + 2) / (dim + mu + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * ((mu + 2) / (mu + 5)) / ((dim + 2) ** 2 + mu))

        # Restart strategy: run multiple CMA-ES instances with different initializations
        # Each instance gets a portion of the budget
        # First, do a quick random search to find a good starting point
        n_random = min(50, budget // 5)
        for i in range(n_random):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        # Now run CMA-ES from the best point found
        # Use the remaining budget
        remaining_budget = budget - calls_made
        if remaining_budget <= lam:
            return best_x, best_f

        # Initialize CMA-ES
        x_mean = best_x.copy()
        f_best = best_f
        sigma = 1.0  # Initial step size
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Weights for the top mu individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        # Main CMA-ES loop
        while calls_made < budget and time.time() < deadline:
            # Generate lambda offspring
            population = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                # Update evolution path for sigma control
                if i == 0:
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * B @ (D * z)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                population.append(y)

            # Evaluate population
            fitnesses = []
            for y in population:
                if time.time() > deadline:
                    fy = np.inf
                else:
                    fy = func(y)
                    calls_made += 1
                fitnesses.append(fy)

            if len(fitnesses) < mu:
                break

            # Sort by fitness
            sorted_indices = np.argsort(fitnesses)
            top_mu_indices = sorted_indices[:mu]
            top_mu_y = np.array([population[i] for i in top_mu_indices])
            top_mu_f = [fitnesses[i] for i in top_mu_indices]

            # Update mean
            new_mean = np.sum(weights[:, None] * top_mu_y, axis=0)
            new_mean = np.clip(new_mean, -5, 5)

            # Update covariance matrix
            y_norm = (top_mu_y - x_mean) / sigma
            delta_mean = (new_mean - x_mean) / sigma

            C_update = np.outer(delta_mean, delta_mean)
            for w, yn in zip(weights, y_norm):
                C_update += w * np.outer(yn, yn)

            # We need to maintain B and D such that C = B diag(D^2) B^T
            # For simplicity, we'll use the full covariance matrix approach
            # Actually, let's use a simpler approach: maintain the covariance matrix directly
            # But that's expensive. Let's use the standard CMA-ES with B and D.
            # For now, let's use a simplified version.

            # Update evolution paths
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_mean

            # Check for stagnation and adapt sigma
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - budget + n_random + 1))) > (chi_n + 1.4) * np.sqrt(1 + (1 - c_sigma) / c_sigma):
                sigma *= np.exp(0.5 * c_sigma * (np.linalg.norm(ps) / chi_n - 1))
            else:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update best
            if top_mu_f[0] < f_best:
                f_best = top_mu_f[0]
                best_f = f_best
                best_x = new_mean.copy()

            x_mean = new_mean

            # Check if we should restart with a smaller sigma or different initialization
            # If sigma is very small, we might be stuck
            if sigma < 1e-6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
