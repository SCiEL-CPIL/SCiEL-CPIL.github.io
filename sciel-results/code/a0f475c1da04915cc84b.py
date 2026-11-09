# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with time guard and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        best_x = np.full(dim, np.nan)
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline - 5.0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial exploration: evaluate several diverse starting points
        rng = np.random.default_rng(42)

        # Try a grid of starting points and random points
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Corners of a smaller box
        for corner in [(0, 0, 0, 0, 0), (1, 1, 1, 1, 1), (-1, -1, -1, -1, -1),
                       (1, -1, 1, -1, 1), (-1, 1, -1, 1, -1)]:
            start_points.append(np.array(corner, dtype=float) * 3.0)
        # Random points
        for _ in range(10):
            start_points.append(rng.uniform(-4, 4, dim))

        # Evaluate initial points
        for sp in start_points:
            if not time_ok() or calls_made >= budget:
                break
            evaluate(sp)

        if calls_made >= budget or not time_ok():
            return best_x, best_f

        # CMA-ES implementation
        def cma_es(x0, sigma0, remaining_budget):
            nonlocal calls_made, best_x, best_f
            dim = self.dim
            mu = max(1, dim // 2)  # number of parents
            lam = 2 * mu  # population size
            cm = np.eye(dim)
            sigma = sigma0
            c_sigma = (mu + 2) / (dim + 4)
            c_c = (4 + dim / 3) / (dim + 4 + 2 * mu)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu)
            c_mu = min(1 - c_1, 2 * ((mu + 0.5) / (dim + 2)) ** 2)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            ps = np.zeros(dim)
            pc = np.zeros(dim)

            pop = []
            # Initial population
            for i in range(lam):
                if not time_ok() or calls_made >= budget:
                    return
                z = rng.standard_normal(dim)
                x_i = x0 + sigma * z
                f_i = evaluate(x_i)
                pop.append((x_i, f_i))

            while time_ok() and calls_made < budget:
                # Sort by fitness
                pop.sort(key=lambda t: t[1])
                m = pop[:mu]

                # Weighted recombination
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mu_eff = 1.0 / np.sum(weights ** 2)

                # Update mean
                x_m = sum(w * x for w, (x, _) in zip(weights, m))

                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (cm @ (x_m - x0))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - lam))) < (1.4 + 2 / (dim + 1)) * chi_n:
                    pc = (1 - c_c) * pc + c_c / sigma * (x_m - x0)

                # Update covariance matrix
                delta_dist = (1 + 2 * (np.dot(ps, ps) / dim > c_sigma / (1 - c_sigma))) * c_sigma
                cm = (1 + c_1 + c_mu - c_1 - c_mu + delta_dist * c_c * (c_c + 2 * c_sigma)) * cm \
                     - c_1 * (np.outer(pc, pc) - cm) \
                     - c_mu * sum(w * np.outer(x - x0, x - x0) for w, (x, _) in zip(weights, m))

                # Make symmetric
                cm = (cm + cm.T) / 2

                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 50.0)

                x0 = x_m

                # Generate new population
                pop = []
                for i in range(lam):
                    if not time_ok() or calls_made >= budget:
                        return
                    z = rng.standard_normal(dim)
                    x_i = x0 + sigma * (cm @ z)
                    f_i = evaluate(x_i)
                    pop.append((x_i, f_i))

                if len(pop) < lam:
                    break

        # Run CMA-ES from the best starting point
        x_start = best_x.copy()
        sigma_start = 2.0
        cma_es(x_start, sigma_start, budget - calls_made)

        # If we still have budget, try a few more local refinements from different points
        if time_ok() and calls_made < budget:
            for _ in range(3):
                if not time_ok() or calls_made >= budget:
                    break
                # Random restart
                x_restart = rng.uniform(-4, 4, dim)
                f_restart = evaluate(x_restart)
                if f_restart < best_f:
                    best_f = f_restart
                    best_x = x_restart.copy()
                # Local search around this point
                sigma_local = 1.0
                x_local = x_restart.copy()
                f_local = f_restart
                for _ in range(min(20, budget - calls_made)):
                    if not time_ok() or calls_made >= budget:
                        break
                    y = np.clip(x_local + sigma_local * rng.standard_normal(dim), -5, 5)
                    fy = evaluate(y)
                    if fy < f_local:
                        x_local, f_local = y, fy
                        sigma_local *= 1.2
                    else:
                        sigma_local *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
