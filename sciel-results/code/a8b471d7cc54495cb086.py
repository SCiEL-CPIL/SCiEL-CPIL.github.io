# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive population size, refined by Nelder-Mead for ridge functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            if calls_made[0] >= budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            calls_made[0] += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Phase 1: CMA-ES with adaptive population size
        # Start with moderate population, shrink as convergence progresses
        
        # Initial population size for CMA-ES
        pop_size = 12  # Start with moderate population for 5D
        min_pop_size = 6
        
        # CMA-ES parameters
        sigma = 1.5  # Initial step size
        x_mean = np.zeros(n)
        
        # Covariance matrix (start identity)
        C = np.eye(n)
        # Eigenvalue decomposition for sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Strategy parameters
        c_sigma = (n + 2) / (n + 5 + 2 * n / (n + 1))
        d_sigma = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + c_sigma
        c_c = (4 + n / n) / (4 + n + n)
        c_1 = 2 / ((n + 1.3) ** 2 + n)
        c_mu = min(1 - c_1, 2 * ((n + 1.) / (n + 4.)) ** 2 - 1)
        
        # Weighted recombination
        mu_eff = int(pop_size / 2)
        weights = np.log(mu_eff + 0.5) - np.log(np.arange(1, mu_eff + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Cumulation parameters
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # History for adaptive population size
        best_f_history = []
        improvement_window = 20  # Check improvement over this many generations
        no_improve_count = 0
        
        gen = 0
        cma_active = True
        
        while calls_made[0] < budget and cma_active:
            if time.time() > deadline:
                break
            
            # Adaptive population size: shrink if converging
            if gen > 0 and len(best_f_history) >= improvement_window:
                recent_best = best_f_history[-improvement_window:]
                if len(recent_best) >= 2:
                    improvement = recent_best[0] - recent_best[-1]
                    if improvement < 1e-6 * (1 + abs(recent_best[0])):
                        no_improve_count += 1
                    else:
                        no_improve_count = 0
                    # Shrink population if stagnating
                    if no_improve_count >= 3 and pop_size > min_pop_size:
                        pop_size = max(min_pop_size, pop_size - 2)
                        no_improve_count = 0
            
            # Sample population
            offspring = []
            for i in range(pop_size):
                if calls_made[0] >= budget or time.time() > deadline:
                    break
                
                # Sample from multivariate normal using current C and sigma
                z = rng.standard_normal(n)
                # Transform using eigendecomposition
                z_transformed = eigvecs @ (np.sqrt(np.maximum(eigvals, 1e-12)) * z)
                x_sample = x_mean + sigma * z_transformed
                x_sample = np.clip(x_sample, -5, 5)
                
                fx = wrapped_func(x_sample)
                offspring.append((x_sample, fx))
            
            if len(offspring) < 2:
                break
            
            # Sort by fitness
            offspring.sort(key=lambda x: x[1])
            
            # Update best
            if offspring[0][1] < best_f:
                best_f = offspring[0][1]
                best_x = offspring[0][0].copy()
            
            # Update mean (weighted recombination of best mu)
            mu = min(mu_eff, len(offspring))
            parents = [offspring[i][0] for i in range(mu)]
            new_mean = np.sum([weights[i] * parents[i] for i in range(mu)], axis=0)
            
            # Update step size (CSA - Cumulative Step Adaptation)
            h_sigma = (np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1) / d_sigma)) <
                       (1 + 2 / (n + 1)) * np.sqrt((mu_eff + 2) / (n + 2)))
            h_sigma = 1.0 if h_sigma else 0.0
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                      (eigvecs.T @ (eigvecs @ (new_mean - x_mean) / sigma))
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update covariance matrix
            y = (new_mean - x_mean) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            
            rank_one = np.outer(y, y)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = (parents[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c_1 * (c_mu - c_1) / c_1 - c_1) * C + c_1 * (rank_one + rank_mu)
            
            # Regularize C to maintain positive semi-definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update state
            x_mean = new_mean
            gen += 1
            
            # Track best fitness history
            best_f_history.append(offspring[0][1])
            if len(best_f_history) > improvement_window + 5:
                best_f_history.pop(0)
            
            # Check for convergence: if simplex is very small, switch to Nelder-Mead
            if sigma < 1e-4 or gen > 80:
                cma_active = False
                break
        
        # Phase 2: Nelder-Mead refinement from best point found
        if calls_made[0] < budget and time.time() < deadline:
            self._nelder_mead_refine(func, wrapped_func, best_x.copy(), best_f, 
                                     calls_made, n, budget, deadline, rng)
        
        return best_x, best_f
    
    def _nelder_mead_refine(self, func, wrapped_func, start_x, start_f, 
                           calls_made, n, budget, deadline, rng):
        """Nelder-Mead refinement from a starting point."""
        nonlocal best_f, best_x
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initialize simplex around start point
        simplex = [start_x.copy()]
        for i in range(n):
            pt = start_x.copy()
            step = 0.3
            if start_x[i] + step > 5:
                pt[i] = start_x[i] - step
            elif start_x[i] - step < -5:
                pt[i] = start_x[i] + step
            else:
                pt[i] = start_x[i] + step
            simplex.append(pt)
        
        # Evaluate initial simplex
        f_values = []
        for pt in simplex:
            if calls_made[0] >= budget or time.time() > deadline:
                break
            f_values.append(wrapped_func(pt))
        
        if len(f_values) < n + 1:
            return
        
        iter_count = 0
        max_iter = 50
        stall_count = 0
        prev_best = best_f
        
        while calls_made[0] < budget and iter_count < max_iter:
            if time.time() > deadline:
                break
            
            # Sort simplex by fitness
            sorted_indices = np.argsort(f_values)
            simplex = [simplex[i] for i in sorted_indices]
            f_values = [f_values[i] for i in sorted_indices]
            
            # Check for convergence
            if f_values[0] < prev_best - 1e-10:
                prev_best = f_values[0]
                stall_count = 0
            else:
                stall_count += 1
                if stall_count > 15:
                    break
            
            x_worst = simplex[-1]
            f_worst = f_values[-1]
            x_best = simplex[0]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = wrapped_func(x_r)
            
            if calls_made[0] >= budget or time.time() > deadline:
                break
            
            if f_r < f_values[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - x_worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = wrapped_func(x_e)
                
                if calls_made[0] >= budget or time.time() > deadline:
                    break
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            elif f_r < f_values[-2]:
                simplex[-1] = x_r
                f_values[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (x_r - centroid)
                
                x_c = np.clip(x_c, -5, 5)
                f_c = wrapped_func(x_c)
                
                if calls_made[0] >= budget or time.time() > deadline:
                    break
                
                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls_made[0] >= budget or time.time() > deadline:
                            break
                        f_values[i] = wrapped_func(simplex[i])
                    
                    if calls_made[0] >= budget or time.time() > deadline:
                        break
                    
                    simplex_size = np.max([np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, n+1)])
                    if simplex_size < 1e-6:
                        break
            
            iter_count += 1
# EVOLVE-BLOCK-END
