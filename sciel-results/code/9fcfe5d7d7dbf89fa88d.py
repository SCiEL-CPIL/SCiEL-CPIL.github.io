"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for Büche-Rastrigin f4 on [-5,5]^5.
    
    Combines:
    1. Structured exploration (integer lattice + random) to find good basins
    2. Cross-Entropy Method (CEM) for population-based basin identification
    3. Multi-start (1+1)-ES with adaptive step size for robust local refinement
    4. Final high-precision polish pass
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        best_x = np.zeros(dim)
        best_f = float('inf')
        budget_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Structured Exploration
        # Sample integer lattice points and random continuous points
        # to identify good basins for local refinement
        
        n_explore = min(80, budget // 4)
        
        # Evaluate integer lattice points (likely near local minima)
        n_lattice = n_explore // 2
        for _ in range(n_lattice):
            if budget_used >= budget or not time_ok():
                break
            pt = rng.integers(-4, 5, size=dim).astype(float)
            pt = np.clip(pt, -5, 5)
            f_val = func(pt)
            budget_used += 1
            update_best(pt, f_val)
        
        # Evaluate random continuous points
        n_random = n_explore - n_lattice
        for _ in range(n_random):
            if budget_used >= budget or not time_ok():
                break
            pt = np.clip(rng.uniform(-5, 5, dim), -5, 5)
            f_val = func(pt)
            budget_used += 1
            update_best(pt, f_val)

        # Phase 2: Cross-Entropy Method (CEM)
        # Iteratively resample from a multivariate normal distribution,
        # updating its mean and covariance to concentrate samples in 
        # the region of the global minimum.
        
        if budget_used < budget and time_ok():
            # CEM parameters
            pop_size = 50
            elite_frac = 0.2  # Keep top 20%
            n_iterations = min(8, (budget - budget_used) // (pop_size + 5))
            
            # Initialize CEM distribution
            mean = best_x.copy()
            # Start with moderate variance
            std = 2.0 * np.ones(dim)
            
            for it in range(n_iterations):
                if budget_used >= budget or not time_ok():
                    break
                
                # Generate population
                pop = mean + std * rng.standard_normal((pop_size, dim))
                pop = np.clip(pop, -5, 5)
                
                # Evaluate population
                pop_f = np.empty(pop_size)
                for i in range(pop_size):
                    if budget_used >= budget or not time_ok():
                        break
                    pop_f[i] = func(pop[i])
                    budget_used += 1
                    update_best(pop[i], pop_f[i])
                
                # If we couldn't evaluate all, break
                if budget_used >= budget:
                    break
                
                # Select elite
                n_elite = max(1, int(pop_size * elite_frac))
                elite_indices = np.argsort(pop_f)[:n_elite]
                elites = pop[elite_indices]
                
                # Update mean and std
                mean = np.mean(elites, axis=0)
                # Use a weighted std to avoid zero variance
                std = np.std(elites, axis=0)
                std = np.maximum(std, 0.1)  # Minimum std to avoid collapse
                
                # Decay std slightly to encourage convergence
                std *= 0.95
                std = np.clip(std, 0.05, 5.0)

        # Phase 3: Multi-Start (1+1)-ES with Adaptive Step Size
        # Use multiple starting points from the exploration phase
        # and run adaptive local search from each
        
        # Collect candidate starting points
        candidates = [best_x.copy()]
        
        # Add a few more random points as additional starting candidates
        n_extra_starts = min(5, max(1, (budget - budget_used) // 10))
        for _ in range(n_extra_starts):
            if budget_used >= budget or not time_ok():
                break
            pt = np.clip(rng.uniform(-5, 5, dim), -5, 5)
            candidates.append(pt)
        
        # Run (1+1)-ES from each candidate
        sigma_init = 1.5
        min_sigma = 0.001
        max_sigma = 10.0
        window_size = 10
        
        for start_pt in candidates:
            if budget_used >= budget or not time_ok():
                break
            
            x = start_pt.copy()
            fx = best_f if np.array_equal(x, best_x) else func(x)
            if not np.array_equal(x, best_x):
                budget_used += 1
                update_best(x, fx)
            
            sigma = sigma_init
            success_window = []
            
            # Run (1+1)-ES from this starting point
            while budget_used < budget and time_ok():
                # Generate trial point
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                budget_used += 1
                
                # Update best and track success
                if fy < fx:
                    x, fx = y, fy
                    success_window.append(1)
                    update_best(x, fx)
                else:
                    success_window.append(0)
                
                # Adapt sigma based on 1/5th success rule
                if len(success_window) >= window_size:
                    success_rate = np.mean(success_window[-window_size:])
                    if success_rate > 0.2:
                        sigma *= 1.05
                    elif success_rate < 0.1:
                        sigma *= 0.95
                    if len(success_window) > window_size:
                        success_window.pop(0)
                    sigma = max(min_sigma, min(sigma, max_sigma))
                
                # If we've converged (very small sigma and no recent improvement),
                # break to try next starting point
                if sigma < 0.01 and len(success_window) >= window_size:
                    if np.mean(success_window[-window_size:]) < 0.05:
                        break

        # Phase 4: Final High-Precision Polish
        # If budget remains, do a fine-grained local search around the best point
        
        remaining = budget - budget_used
        if remaining > 10 and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.05  # Small step size for fine refinement
            min_sigma = 1e-5
            max_sigma = 0.5
            window_size = 5
            success_window = []
            
            for _ in range(remaining):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                budget_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_window.append(1)
                    update_best(x, fx)
                else:
                    success_window.append(0)
                
                # Adapt sigma
                if len(success_window) >= window_size:
                    success_rate = np.mean(success_window[-window_size:])
                    if success_rate > 0.3:
                        sigma *= 1.1
                    elif success_rate < 0.1:
                        sigma *= 0.8
                    if len(success_window) > window_size:
                        success_window.pop(0)
                    sigma = max(min_sigma, min(sigma, max_sigma))
        
        return best_x, best_f
# EVOLVE-BLOCK-END
