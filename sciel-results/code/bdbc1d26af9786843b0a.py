"""CMA-ES based optimizer for BBOB f22 (Gallagher 21 peaks) with Halton initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with Halton-based initial population and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return None
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            fx = float(func(x))
            if not np.isfinite(fx):
                fx = 1e10
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def halton_sequence(n, dim, base_start=2):
            """Generate Halton sequence points in [0,1]^dim."""
            points = np.zeros((n, dim))
            for d in range(dim):
                base = base_start + d
                for i in range(n):
                    f = 1.0
                    r = 0.0
                    idx = i + 1
                    while idx > 0:
                        idx, rem = divmod(idx, base)
                        f /= base
                        r += rem * f
                    points[i, d] = r
            return points

        # Phase 1: Halton space-filling exploration for good initial coverage
        n_halton = min(60, max(30, budget // 8))
        
        # Add center point first
        x0 = np.zeros(dim)
        fx0 = eval_func(x0)
        if fx0 is None:
            return best_x, best_f

        # Halton points mapped to [-5, 5]^dim
        halton_pts = halton_sequence(n_halton - 1, dim)
        for i in range(n_halton - 1):
            if time.time() > deadline:
                break
            if calls_made >= budget - 50:
                break
            x = halton_pts[i] * 10.0 - 5.0  # map [0,1] to [-5,5]
            x = np.clip(x, -5.0, 5.0)
            fx = eval_func(x)
            if fx is None:
                break

        if best_x is None:
            return best_x, best_f

        # Phase 2: CMA-ES optimization
        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # population size
        sigma = 1.0  # initial step size
        max_iter = 200  # maximum CMA-ES iterations

        # Initialize population around best point
        rng = np.random.default_rng(42)
        population = np.zeros((pop_size, dim))
        fitness = np.zeros(pop_size)
        
        for i in range(pop_size):
            if time.time() > deadline:
                break
            if calls_made >= budget - 10:
                break
            # Generate candidate around best_x with some diversity
            x = best_x + sigma * rng.standard_normal(dim)
            x = np.clip(x, -5.0, 5.0)
            fx = eval_func(x)
            if fx is None:
                break
            population[i] = x
            fitness[i] = fx

        # CMA-ES main loop
        mean = best_x.copy()
        C = np.eye(dim)  # covariance matrix
        step_size = sigma
        success_count = 0
        total_steps = 0
        last_improvement_iter = 0

        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            if calls_made >= budget - 10:
                break

            # Generate population
            for i in range(pop_size):
                if calls_made >= budget - 10:
                    break
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                # Cholesky decomposition for sampling from covariance
                try:
                    L = np.linalg.cholesky(C)
                    x = mean + step_size * (L @ z)
                except:
                    x = mean + step_size * z
                
                x = np.clip(x, -5.0, 5.0)
                fx = eval_func(x)
                if fx is None:
                    break
                population[i] = x
                fitness[i] = fx

            if calls_made >= budget - 10:
                break

            # Sort by fitness
            idx = np.argsort(fitness)
            population = population[idx]
            fitness = fitness[idx]

            # Update best
            if fitness[0] < best_f:
                best_f = fitness[0]
                best_x = population[0].copy()
                last_improvement_iter = iteration

            # Update mean using weighted average of top half
            lambda_weight = pop_size // 2
            weights = np.zeros(pop_size)
            weights[:lambda_weight] = 1.0 / lambda_weight
            mean = np.sum(weights[:, None] * population, axis=0)
            mean = np.clip(mean, -5.0, 5.0)

            # Update covariance matrix
            # Simple rank-mu update
            diff = population[:lambda_weight] - mean
            C_new = np.zeros((dim, dim))
            for i in range(lambda_weight):
                C_new += weights[i] * np.outer(diff[i], diff[i])
            
            # Combine with old covariance for stability
            C = 0.7 * C + 0.3 * C_new
            # Ensure positive semi-definite
            C = (C + C.T) / 2
            # Regularize
            C += 1e-10 * np.eye(dim)

            # Adapt step size based on success rate
            if fitness[0] < best_f + 1e-8:
                success_count += 1
            total_steps += 1
            
            if total_steps >= 10:
                rate = success_count / total_steps
                if rate > 0.3:
                    step_size *= 1.1
                elif rate < 0.1:
                    step_size *= 0.9
                success_count = 0
                total_steps = 0
            
            step_size = np.clip(step_size, 1e-12, 5.0)

            # Early stop if converged
            if step_size < 1e-10 or (iteration - last_improvement_iter > 30 and step_size < 0.1):
                break

        # Phase 3: Fine local search with small step size ES
        if best_x is not None and calls_made < budget - 5:
            remaining_fine = budget - calls_made - 5
            if remaining_fine > 5:
                # (1+1)-ES with very small step size
                rng_fine = np.random.default_rng(12345)
                x = best_x.copy()
                fx = best_f
                sigma_fine = 0.01
                for _ in range(remaining_fine):
                    if time.time() > deadline:
                        break
                    if calls_made >= budget - 2:
                        break
                    z = rng_fine.standard_normal(dim)
                    x_new = x + sigma_fine * z
                    x_new = np.clip(x_new, -5.0, 5.0)
                    fx_new = eval_func(x_new)
                    if fx_new is None:
                        break
                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        sigma_fine *= 0.95
                    sigma_fine = np.clip(sigma_fine, 1e-14, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
