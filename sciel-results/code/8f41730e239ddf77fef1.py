"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-stage search: Latin Hypercube bracketing + CMA-ES exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track global best
        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget or time.time() - start_time > time_limit:
                return np.inf
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            calls_used += 1
            try:
                val = func(x_clipped)
                return float(val)
            except Exception:
                return np.inf

        # Phase 1: Coarse Latin Hypercube sampling to rapidly bracket the basin
        num_initial = min(20, budget // 10)
        initial_points = []

        levels = np.array([-3.5, 0.0, 3.5])
        
        # First: center
        initial_points.append(np.zeros(dim))
        
        # Then: axis-aligned extremes
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 3.5
                initial_points.append(p)
        
        # Fill remaining with stratified random points
        while len(initial_points) < num_initial:
            p = np.array([levels[rng.integers(0, 3)] + rng.uniform(-1.0, 1.0) for _ in range(dim)])
            p = np.clip(p, -4.5, 4.5)
            initial_points.append(p)

        # Evaluate initial points
        local_bests = []
        for p in initial_points:
            if calls_used >= budget or time.time() - start_time > time_limit:
                break
            f = safe_call(p)
            if f < best_f:
                best_f = f
                best_x = p.copy()
            local_bests.append((f, p.copy()))

        # Sort by fitness
        local_bests.sort(key=lambda x: x[0])

        # Phase 2: CMA-ES from the best starting point
        if calls_used >= budget or time.time() - start_time > time_limit:
            return best_x, best_f

        f0, x0 = local_bests[0]

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # ~15 for dim=5
        sigma = 0.5  # Initial step size
        max_iter = (budget - calls_used) // pop_size

        # Population
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        step_size = 1.0  # Not used directly, C handles scaling

        # Initial population
        mean = x0.copy()
        population = [mean + sigma * rng.standard_normal(dim) for _ in range(pop_size)]
        population[0] = mean  # First individual is the mean

        for iteration in range(max_iter):
            if calls_used >= budget or time.time() - start_time > time_limit:
                break

            # Evaluate population
            fitnesses = []
            for ind in population:
                f = safe_call(ind)
                fitnesses.append(f)
                if f < best_f:
                    best_f = f
                    best_x = ind.copy()

            # Sort by fitness
            order = np.argsort(fitnesses)
            population = [population[i] for i in order]
            fitnesses = [fitnesses[i] for i in order]

            # Update mean
            new_mean = np.sum([weights[i] * population[i] for i in range(mu)], axis=0)
            
            # Update evolution path for sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C @ (new_mean - mean))
            
            # Update evolution path for C
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (iteration + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (new_mean - mean)
            
            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_c) * C + c_1 * (np.outer(p_c, p_c) - delta_h * C)
            C += c_mu * np.sum([weights[i] * np.outer(population[i] - mean, population[i] - mean) for i in range(mu)], axis=0)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            mean = new_mean

            # Generate new population
            population = []
            for i in range(pop_size):
                # Use Cholesky decomposition for sampling
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                z = rng.standard_normal(dim)
                individual = mean + sigma * L @ z
                population.append(individual)
            population[0] = mean  # First individual is the mean

        return best_x, best_f
# EVOLVE-BLOCK-END
