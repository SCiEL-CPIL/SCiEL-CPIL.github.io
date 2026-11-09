"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        total_calls = 0

        def count():
            nonlocal total_calls
            total_calls += 1

        # Try multiple restarts with different strategies
        # First, do a coarse random search to find a good starting point
        # Then apply CMA-ES-like local search

        # Phase 1: Random sampling to get a decent initial point
        n_initial = min(50, self.budget // 10)
        best_init_x = None
        best_init_f = np.inf
        for i in range(n_initial):
            if total_calls >= self.budget - 10:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            count()
            if fx < best_init_f:
                best_init_f = fx
                best_init_x = x.copy()
                best_x = x.copy()
                best_f = fx

        if best_init_x is None:
            x = np.zeros(self.dim)
            fx = func(x)
            count()
            best_x = x
            best_f = fx
            best_init_x = x
            best_init_f = fx

        # Phase 2: CMA-ES-like local search from the best point
        # Use a population-based approach with adaptive sigma
        remaining = self.budget - total_calls
        if remaining <= 0:
            return best_x, best_f

        # CMA-ES parameters
        lam = max(4, self.dim + int(3 * np.log(self.dim)))  # population size
        sigma = 1.0
        C = np.eye(self.dim)
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        c_sigma = (lam + 2) / (self.dim + lam + 5)
        c_c = (4 + lam / self.dim) / (self.dim + 4 + 2 * lam / self.dim)
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        c_1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2) / ((self.dim + 2) ** 2 + mu_eff))

        x_mean = best_init_x.copy()
        f_mean = best_init_f

        while total_calls < self.budget:
            # Generate population
            pop = []
            for i in range(lam):
                if total_calls >= self.budget:
                    break
                z = rng.standard_normal(self.dim)
                # Transform using C
                y = x_mean + sigma * (C @ z)
                y = np.clip(y, -5, 5)
                pop.append((y, None))

            # Evaluate
            for i, (y, _) in enumerate(pop):
                if total_calls >= self.budget:
                    break
                fy = func(y)
                count()
                pop[i] = (y, fy)

            if not pop:
                break

            # Sort by fitness
            pop.sort(key=lambda t: t[1])
            
            # Update best
            if pop[0][1] < best_f:
                best_f = pop[0][1]
                best_x = pop[0][0].copy()

            # Update mean
            x_new = np.zeros(self.dim)
            for i in range(mu):
                x_new += weights[i] * pop[i][0]
            
            # Update path accumulators
            sigma_old = sigma
            c_sigma_sqrt = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff / self.dim)
            ps = (1 - c_sigma) * ps + c_sigma_sqrt * (C @ (x_new - x_mean) / sigma)
            h_sig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1 + 2 / (self.dim + 1)) * np.sqrt(self.dim + 1 - 2) else 0
            
            c_c_sqrt = np.sqrt(c_c * (2 - c_c) * mu_eff / self.dim)
            pc = (1 - c_c) * pc + c_c_sqrt * h_sig * (x_new - x_mean) / sigma

            # Update covariance matrix
            delta_h = (1 - h_sig) * c_c * (2 - c_c)
            diff = x_new - x_mean
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) - C) + c_mu * sum(weights[i] * np.outer(pop[i][0] - x_mean, pop[i][0] - x_mean) for i in range(mu))

            # Update sigma
            chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(1e-8, min(sigma, 10.0))

            x_mean = x_new
            f_mean = pop[0][1]

            # Check for stagnation - if no improvement for a while, restart
            if pop[0][1] >= best_f and total_calls > self.budget // 2:
                # Small perturbation
                x_mean = x_mean + 0.5 * rng.standard_normal(self.dim)
                x_mean = np.clip(x_mean, -5, 5)
                sigma = max(sigma, 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
