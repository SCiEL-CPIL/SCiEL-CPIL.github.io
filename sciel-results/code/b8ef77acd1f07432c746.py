"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """CMA-ES with adaptive restarts and fast initial screening."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Global best tracking
        global_best_x = None
        global_best_f = float('inf')
        total_calls = 0
        start_time = time.time()
        max_time = 280.0  # Safety margin for 330s limit (evaluator kills at 330)

        def can_continue():
            return total_calls < budget and (time.time() - start_time) < max_time

        # --- Phase 1: Fast Initial Screening ---
        # Goal: Get a decent starting point quickly to boost early AOCC.
        # Use a coarse grid or random sampling with a small budget.
        n_screen = 20
        # Use a mix of random and grid-like points for coverage
        # Since dim=5, a full grid is too expensive, so random with some structure
        for i in range(n_screen):
            if not can_continue():
                break
            # Mix of uniform and slightly structured
            if i % 2 == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                # Simplex-like or offset random
                x = rng.uniform(-4, 4, dim)
            fx = func(x)
            total_calls += 1
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()

        # --- Phase 2: CMA-ES with Restarts ---
        # Parameters
        pop_size = 4 + int(math.floor(3 * math.log(dim)))  # ~10 for dim=5
        # Ensure pop_size is odd? CMA-ES usually works with any, but let's stick to standard.
        # For dim=5, pop_size = 4 + 3*1.6 = 8.8 -> 11? 
        # Standard formula: lambda = 4 + floor(3 * log(n)). For n=5, log(5)=1.6, 3*1.6=4.8, floor=4. 4+4=8.
        # Let's use 9 or 10.
        pop_size = 9
        
        # Restart logic:
        # We will run CMA-ES for a certain number of generations or until stagnation.
        # Then restart from the current global best with a slightly different strategy or just fresh CMA-ES.
        
        max_generations_per_run = 20
        stagnation_threshold = 0.01  # If improvement is less than this over last 10 gens, restart
        
        run_id = 0
        while can_continue():
            run_id += 1
            # Initialize CMA-ES state
            # Start from global best
            x_mean = global_best_x.copy()
            sigma = 1.0  # Initial step size. Could be adaptive based on distance to bounds?
            
            # CMA-ES parameters
            mu = pop_size // 2  # Number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / (dim**1.7 + 10)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim**2 + 18))
            
            # Covariance matrix
            C = np.eye(dim)
            # Inverse square root of C (initially identity)
            C_sqrt_inv = np.eye(dim)
            # Eigenvalues and eigenvectors for C (for efficient update)
            # We will update C using rank-1 and rank-mu updates
            # For simplicity and speed in dim=5, we can just maintain C and compute sqrt each step?
            # Or use the standard CMA-ES update rules.
            
            # Let's implement a simplified but robust CMA-ES update.
            # Since dim is small (5), full matrix operations are cheap.
            
            # Normalization of p_sigma
            chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            # History
            p_c = np.zeros(dim)
            p_sigma = np.zeros(dim)
            
            best_f_run = global_best_f
            prev_best_f = global_best_f
            no_improve_count = 0
            
            # Generations
            for gen in range(max_generations_per_run):
                if not can_continue():
                    break
                
                # Generate population
                # Need C^0.5. We can use eigen decomposition each step or maintain it.
                # For dim=5, eigen decomposition is fast.
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small eigenvalues
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                # Sample
                pop = np.array([x_mean + sigma * C_sqrt @ rng.standard_normal(dim) for _ in range(pop_size)])
                # Clip to bounds? CMA-ES doesn't handle bounds well natively.
                # The task says func clips. So we can just pass raw values, but 
                # it's better to stay inside to avoid wasted calls if func penalizes?
                # Task: "func(x) clips a finite length-5 vector to the box". 
                # So calling with out-of-bounds is safe, but might be less efficient 
                # if the function is defined differently outside? No, it clips.
                # So we can let CMA-ES go out of bounds, but it might be better to 
                # reflect or clip. Clipping distorts the distribution.
                # Let's clip to be safe and efficient within the defined domain.
                pop = np.clip(pop, -5, 5)
                
                # Evaluate
                fvals = np.array([func(x) for x in pop])
                total_calls += pop_size
                
                # Update global best
                idx_min = np.argmin(fvals)
                if fvals[idx_min] < global_best_f:
                    global_best_f = fvals[idx_min]
                    global_best_x = pop[idx_min].copy()
                    best_f_run = fvals[idx_min]
                    
                if best_f_run < prev_best_f:
                    prev_best_f = best_f_run
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if no_improve_count > 5:
                    break # Stagnation, restart soon
                    
                # Sort population
                sorted_indices = np.argsort(fvals)
                # Take top mu
                parents = pop[sorted_indices[:mu]]
                weights_sorted = weights
                
                # Weighted mean
                x_new_mean = np.sum(weights_sorted[:, np.newaxis] * parents, axis=0)
                
                # Update covariance matrix
                # Rank-1 update
                y_mean = C_sqrt_inv @ (x_new_mean - x_mean)
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * d_sigma * y_mean
                h_sigma = (np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) / chi_n) < (1.4 + 2 / (dim + 1))
                
                p_c = (1 - c_c) * p_c + h_sigma * c_c * (x_new_mean - x_mean) / sigma
                
                C = (1 + c_1 * h_sigma * (2 - c_1) - c_1 - c_mu) * C \
                    + c_1 * np.outer(p_c, p_c) / sigma**2 \
                    + c_mu * np.sum(weights_sorted[:, np.newaxis] * np.array([np.outer(C_sqrt_inv @ (parents[i] - x_mean), C_sqrt_inv @ (parents[i] - x_mean)) for i in range(mu)]), axis=0)
                    
                # Update sigma
                sigma *= math.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update mean
                x_mean = x_new_mean
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                eigvals, _ = np.linalg.eigh(C)
                if np.any(eigvals < 1e-12):
                    # Reset if singular
                    break

            # End of CMA-ES run
            # If we ran out of time or budget, exit
            if not can_continue():
                break

            # Restart logic:
            # If we stagnated, restart.
            # We could also try a different initial sigma or start point.
            # For now, just restart from global best with fresh CMA-ES.
            # To avoid identical restarts, we can perturb the start point slightly.
            # But CMA-ES will adapt.
            # Let's reduce pop_size if we are very close? No, keep it simple.

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
