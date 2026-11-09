import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        self.func = func
        self.deadline = self.start_time + 290.0
        self.best_x = None
        self.best_f = float('inf')
        self.dim = self.dim
        self.budget = self.budget

        def wrapped_func(x):
            if self.call_count >= self.budget:
                return float('inf')
            if time.time() > self.deadline:
                return float('inf')
            
            x = np.asarray(x)
            if x.shape != (self.dim,):
                x = np.reshape(x, self.dim)
            
            self.call_count += 1
            f_val = float(func(x))
            
            if np.isfinite(f_val) and f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
            
            return f_val

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Multi-Island CMA-ES Strategy
        # Structural change: Instead of sequential restarts, we run multiple CMA-ES instances
        # in parallel (conceptually, sequentially in code but with shared memory for 'crossing').
        # The 'cross' operator here is a weighted recombination of the best individuals from
        # all islands to seed a new island when one stagnates. This tests if global information
        # sharing helps escape local minima in Gallagher 101.

        n_islands = 3
        budget_per_island = budget // n_islands
        # Allocate extra budget to the first island for initial exploration
        island_budgets = [budget_per_island + (budget - n_islands * budget_per_island)] + [budget_per_island] * (n_islands - 1)

        # Store the best individual from each island
        island_best_x = [None] * n_islands
        island_best_f = [float('inf')] * n_islands
        
        # Store recent bests for crossing
        all_best_x = []
        all_best_f = []

        for i in range(n_islands):
            if self.call_count >= budget or time.time() > self.deadline:
                break

            # Initialize island i
            # If i == 0, start at center.
            # If i > 0, use a 'cross' of previous bests to initialize the mean.
            # This is the structural change: using global info to seed local search.
            if i == 0:
                mean = np.zeros(dim)
            else:
                # Cross: Weighted average of the best points found so far from other islands
                # This aims to jump to a potentially better basin or a region between basins.
                if all_best_x:
                    # Weight by inverse fitness (approx) or just uniform average for simplicity
                    # Let's use a random weighted average to introduce diversity
                    weights = rng.dirichlet(np.ones(len(all_best_x)))
                    mean = np.average(np.array(all_best_x), axis=0, weights=weights)
                else:
                    mean = rng.uniform(-4, 4, dim)
            
            # Initial sigma: Large enough to explore, but not too large to miss the global min
            sigma = 2.0
            lam = 8 # Population size
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff)))

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            C = np.eye(dim)

            # Run CMA-ES for this island
            local_best_x = mean.copy()
            local_best_f = float('inf')
            
            # Evaluate initial mean
            if self.call_count < budget:
                f_init = wrapped_func(mean)
                if np.isfinite(f_init):
                    local_best_f = f_init
                    local_best_x = mean.copy()
                    island_best_x[i] = mean.copy()
                    island_best_f[i] = local_best_f
                    all_best_x.append(mean.copy())
                    all_best_f.append(local_best_f)

            gen = 0
            while self.call_count < budget and time.time() < self.deadline:
                if budget - self.call_count < lam:
                    break
                
                # Decompose C
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                L = eigvecs @ np.diag(np.sqrt(eigvals))

                # Generate population
                z_pop = rng.standard_normal((lam, dim))
                y_pop = z_pop @ L.T * sigma
                x_pop = mean + y_pop
                
                # Evaluate
                f_pop = np.zeros(lam)
                for j in range(lam):
                    if self.call_count >= budget or time.time() > self.deadline:
                        break
                    f_pop[j] = wrapped_func(x_pop[j])
                
                if time.time() > self.deadline:
                    break

                # Update bests
                order = np.argsort(f_pop)
                x_pop_sorted = x_pop[order]
                f_pop_sorted = f_pop[order]
                
                if f_pop_sorted[0] < local_best_f:
                    local_best_f = f_pop_sorted[0]
                    local_best_x = x_pop_sorted[0].copy()
                    island_best_x[i] = x_pop_sorted[0].copy()
                    island_best_f[i] = local_best_f
                    # Update global list for future crossings
                    # Keep only top 5 to avoid memory bloat and bias towards old points
                    if len(all_best_x) < 5:
                        all_best_x.append(x_pop_sorted[0].copy())
                        all_best_f.append(local_best_f)
                    else:
                        # Replace the worst in the list
                        worst_idx = np.argmax(all_best_f)
                        if local_best_f < all_best_f[worst_idx]:
                            all_best_x[worst_idx] = x_pop_sorted[0].copy()
                            all_best_f[worst_idx] = local_best_f

                # CMA-ES Update
                new_mean = np.dot(weights, x_pop_sorted[:mu])
                y_selected = (x_pop_sorted[:mu] - mean) / sigma
                H = np.sum(weights[:, None] * (y_selected[:, :, None] * y_selected[:, None, :]), axis=0)
                y_mean = (new_mean - mean) / sigma
                
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n
                h_sigma = 1.0 if h_sigma else 0.0
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                
                C = (1 - c_1 - c_mu) * C + c_1 * (p_c[:, None] * p_c[None, :] + (h_sigma - 1) * C) + c_mu * H
                C = (C + C.T) / 2
                C += 1e-12 * np.eye(dim)
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                mean = new_mean
                gen += 1

        if self.best_x is None:
            self.best_x = np.zeros(dim)
            if self.call_count < budget:
                self.best_f = wrapped_func(self.best_x)
            else:
                self.best_f = 0.0

        return self.best_x, self.best_f
