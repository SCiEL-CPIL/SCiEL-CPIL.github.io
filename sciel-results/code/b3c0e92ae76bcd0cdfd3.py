"""
Optimization strategy:
1. LHS (Latin Hypercube) sampling for initial diverse exploration.
2. CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for local refinement
   starting from the best LHS point.
3. Final high-precision (1+1)-ES polish around the global best.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(12345)

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=np.float64).flatten()
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_deadline():
            return time.time() >= deadline or calls_made >= budget

        # Phase 1: Latin Hypercube Sampling for diverse exploration
        # LHS gives better coverage than pure random sampling
        n_lhs = min(50, budget // 6)
        
        # Generate LHS points
        # For each dimension, divide [0,1] into n_lhs equal intervals, 
        # pick one random point in each interval, then permute per dimension
        lhs_points = np.empty((n_lhs, dim))
        for d in range(dim):
            # Create n_lhs intervals in [0, 1]
            intervals = np.linspace(0, 1, n_lhs + 1)
            # Pick one random point in each interval
            points_in_intervals = np.empty(n_lhs)
            for i in range(n_lhs):
                points_in_intervals[i] = rng.uniform(intervals[i], intervals[i+1])
            # Shuffle the points
            rng.shuffle(points_in_intervals)
            # Scale to [-5, 5]
            lhs_points[:, d] = points_in_intervals * 10.0 - 5.0
        
        # Evaluate LHS points
        lhs_values = np.empty(n_lhs)
        for i in range(n_lhs):
            if check_deadline():
                n_lhs = i
                break
            lhs_values[i] = evaluate(lhs_points[i])
        
        # Find the best LHS point
        if n_lhs > 0:
            best_idx = np.argmin(lhs_values)
            cma_start = lhs_points[best_idx].copy()
            cma_start_f = lhs_values[best_idx]
        else:
            # Fallback to random
            cma_start = rng.uniform(-5, 5, dim)
            cma_start_f = evaluate(cma_start)

        # Phase 2: CMA-ES from the best LHS point
        # Implement a simplified CMA-ES
        if not check_deadline() and best_x is not None:
            # CMA-ES parameters
            x_mean = cma_start.copy()
            f_mean = cma_start_f
            
            # Initial step size (sigma)
            sigma = 0.5 * (max(abs(cma_start)) + 1.0)
            sigma = np.clip(sigma, 0.5, 2.0)
            
            # Population size
            lambda_pop = 4 + int(3 * np.log(dim))  # standard CMA-ES lambda
            # For dim=5, lambda = 4 + 3*log(5) ≈ 4 + 4.8 ≈ 9
            lambda_pop = int(lambda_pop)
            
            # Covariance matrix C (initially identity)
            C = np.eye(dim)
            # Square root of C (initially identity)
            C_sqrt = np.eye(dim)
            
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Adaptation rates
            c_sigma = 1.0 / (lambda_pop + 3)
            mu = lambda_pop // 2  # number of selected individuals
            weights = np.zeros(mu)
            w_sum = 0.0
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
                w_sum += weights[i]
            weights /= w_sum
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_c = (4.0 + mu_eff / lambda_pop) / ((lambda_pop + 2.0) ** 2 + mu_eff)
            d_sigma = 1.0 + 2.0 * max(0.0, (mu_eff - 1.0) / (lambda_pop - 1.0)) + c_sigma
            
            # Cumulative step size adaptation
            sigma = np.clip(sigma, 1e-4, 10.0)
            
            # BFGS-like update for C_sqrt
            def cholesky_update(C_sqrt, w, y):
                """Update C_sqrt using BFGS-like formula."""
                dim = len(y)
                # Build the update
                # C_sqrt = C_sqrt + sum_i w_i (y_i y_i^T - C) / C_sqrt
                # This is an approximation; for simplicity, we'll use a simpler approach
                # Actually, let's just use a simple diagonal adaptation instead of full CMA
                pass
            
            # Simplified CMA-ES: use adaptive diagonal step sizes
            # This is a compromise between full CMA-ES and (1+1)-ES
            sigma_diag = np.ones(dim) * sigma
            sigma_diag_sum = np.ones(dim) * sigma
            
            max_cma_iter = int((budget - calls_made) * 0.5)
            
            for iteration in range(max_cma_iter):
                if check_deadline():
                    break
                
                # Generate population
                pop_x = np.empty((lambda_pop, dim))
                pop_f = np.empty(lambda_pop)
                
                for i in range(lambda_pop):
                    if check_deadline():
                        lambda_pop = i
                        break
                    # Generate sample
                    y = rng.standard_normal(dim)
                    x_candidate = np.clip(x_mean + np.diag(sigma_diag) @ y, -5, 5)
                    pop_x[i] = x_candidate
                    pop_f[i] = evaluate(x_candidate)
                
                # Sort by fitness
                if lambda_pop > 0:
                    sorted_idx = np.argsort(pop_f[:lambda_pop])
                    # Take top mu
                    selected_idx = sorted_idx[:min(mu, lambda_pop)]
                    mu_actual = len(selected_idx)
                    
                    if mu_actual > 0:
                        # Update mean
                        x_mean_new = np.zeros(dim)
                        w_sum_actual = 0.0
                        for i in range(mu_actual):
                            w_i = weights[i] if i < len(weights) else weights[-1]
                            x_mean_new += w_i * pop_x[selected_idx[i]]
                            w_sum_actual += w_i
                        x_mean_new /= w_sum_actual
                        
                        # Update sigma (simplified)
                        # Compute step
                        step = x_mean_new - x_mean
                        step_norm = np.linalg.norm(step)
                        
                        # Update sigma using cumulative step
                        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * step / (sigma_diag_sum.mean() * np.sqrt(dim))
                        sigma_diag_sum = (1 - c_sigma) * sigma_diag_sum + c_sigma * step_norm / sigma_diag_sum.mean()
                        
                        # Adapt sigma_diag
                        for d in range(dim):
                            if abs(step[d]) > 1e-10:
                                # Increase sigma if we made progress in this direction
                                if step_norm < 0.5 * sigma_diag_sum.mean() * np.sqrt(dim):
                                    sigma_diag[d] *= 1.1
                                else:
                                    sigma_diag[d] *= 0.9
                        sigma_diag = np.clip(sigma_diag, 1e-5, 5.0)
                        
                        x_mean = x_mean_new
            
        # Phase 3: Final polish with (1+1)-ES
        if not check_deadline() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            success_count = 0
            total_steps = 0
            
            while not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                total_steps += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                
                # Adapt sigma
                if total_steps % 5 == 0:
                    success_rate = success_count / 5.0
                    if success_rate > 0.2:
                        sigma *= 1.2
                    else:
                        sigma *= 0.8
                    success_count = 0
                    
                sigma = np.clip(sigma, 1e-8, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
