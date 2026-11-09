import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        n = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = np.zeros(n)
        best_f = np.inf

        def do_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Phase 1: Latin Hypercube Sampling for broad exploration
        # LHS provides better space-filling properties than uniform random
        num_lhs = 60
        lhs_points = []
        for i in range(n):
            # Create n intervals and sample one point from each
            intervals = np.linspace(-5.0, 5.0, n + 1)
            samples = rng.uniform(intervals[:-1], intervals[1:])
            # Permute to decorrelate dimensions
            perm = rng.permutation(n)
            lhs_points.append(samples[perm])
        lhs_points = np.array(lhs_points).T  # shape (n, n)
        
        # Actually, let's do proper LHS: sample one point per dimension per stratum
        lhs_samples = np.zeros((num_lhs, n))
        for d in range(n):
            strata = np.linspace(-5.0, 5.0, num_lhs + 1)
            pts = rng.uniform(strata[:-1], strata[1:])
            order = rng.permutation(num_lhs)
            lhs_samples[:, d] = pts[order]

        for p in lhs_samples:
            if not time_ok() or calls_made >= total_budget:
                break
            do_call(p)

        # Get the best point from LHS
        # We need to track which point gave the best value
        # Since do_call updates best_x, best_x is now the best LHS point

        # Phase 2: Multi-start CMA-ES with adaptive restarts
        # CMA-ES is well-suited for black-box continuous optimization
        
        def cma_es(start_x, budget, seed_offset=0):
            nonlocal calls_made, best_f, best_x
            rng_local = np.random.default_rng(42 + seed_offset)
            
            # CMA-ES parameters
            pop_size = 4 + int(3 * math.log(n))  # lambda
            sigma = 0.5  # Initial step size
            x_mean = start_x.copy()
            
            # Covariance matrix
            C = np.eye(n)
            # Eigenvalues for C
            eigvals = np.ones(n)
            # Inverse square root of C
            C_inv_sqrt = np.eye(n)
            
            # Learning rates
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            
            # Evolution path
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Normalization constants
            chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            # Budget management
            local_calls = 0
            max_iter = budget // (pop_size + 1)
            
            for iteration in range(max_iter):
                if not time_ok() or calls_made >= total_budget:
                    break
                
                # Generate population
                offspring = []
                for i in range(pop_size):
                    z = rng_local.standard_normal(n)
                    # Apply CMA-ES transformation: x = mean + sigma * C^(1/2) * z
                    # For simplicity, use diagonal approximation if full C is expensive
                    # But let's do proper CMA-ES with Cholesky
                    # C^(1/2) via Cholesky decomposition
                    try:
                        L = np.linalg.cholesky(C)
                        y = L @ z
                    except:
                        y = z  # Fallback
                    x_i = x_mean + sigma * y
                    x_i = np.clip(x_i, -5.0, 5.0)
                    offspring.append(x_i)
                
                # Evaluate offspring
                values = []
                for x_i in offspring:
                    f_i = do_call(x_i)
                    local_calls += 1
                    values.append(f_i)
                    if not time_ok() or calls_made >= total_budget:
                        break
                
                if len(values) < pop_size:
                    break
                
                values = np.array(values)
                
                # Sort by fitness
                order = np.argsort(values)
                offspring = [offspring[i] for i in order]
                values = values[order]
                
                # Update mean
                x_new_mean = np.zeros(n)
                for i in range(mu):
                    x_new_mean += weights[i] * offspring[i]
                
                # Update evolution path p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ (x_new_mean - x_mean))
                
                # Update evolution path p_c
                h_sigma = (np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                p_c = (1 - c_c) * p_c + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma
                
                # Update covariance matrix
                delta = 1 + 2 * min((1 - h_sigma) * c_c * (2 - c_c), 1.0) * mu_eff / n
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    diff = (offspring[i] - x_mean) / sigma
                    rank_mu += weights[i] * np.outer(diff, diff)
                
                C = (1 - c_1 - c_mu) * C + c_1 * delta * rank_one + c_mu * rank_mu
                
                # Update sigma
                sigma *= math.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update mean
                x_mean = x_new_mean
                
                # Periodically recompute C_inv_sqrt
                if iteration % 10 == 0:
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.clip(eigvals, 1e-12, None)
                        C_inv_sqrt = (eigvecs * (1 / np.sqrt(eigvals))) @ eigvecs.T
                    except:
                        pass
                
                # Check convergence
                if np.std(values[:mu]) < 1e-10:
                    break
                
                # Adapt sigma if stuck
                if iteration > 20 and sigma < 1e-6:
                    break

            return best_f

        # Multi-start CMA-ES
        # Select top 3 diverse points from LHS
        # We'll just use the best point and a couple of random perturbations
        num_starts = 3
        remaining_budget = total_budget - calls_made
        
        if remaining_budget > 50:
            # Start 1: best point found so far
            budget_per_start = remaining_budget // num_starts
            
            # Start from best_x
            cma_es(best_x, budget_per_start, seed_offset=1)
            
            # Start 2: best_x with small perturbation
            if time_ok() and calls_made < total_budget:
                start2 = best_x + rng.normal(0, 0.5, n)
                start2 = np.clip(start2, -5.0, 5.0)
                cma_es(start2, budget_per_start, seed_offset=2)
            
            # Start 3: another random point
            if time_ok() and calls_made < total_budget:
                start3 = rng.uniform(-4.0, 4.0, n)
                cma_es(start3, budget_per_start, seed_offset=3)
        
        # Final local refinement using Nelder-Mead from best point
        if time_ok() and calls_made < total_budget and best_f < np.inf:
            # Simple local search
            step = 0.1
            for _ in range(20):
                if not time_ok() or calls_made >= total_budget:
                    break
                improved = False
                for j in range(n):
                    for direction in [-1, 1]:
                        x_try = best_x.copy()
                        x_try[j] += direction * step
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = do_call(x_try)
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break

        return best_x, best_f
