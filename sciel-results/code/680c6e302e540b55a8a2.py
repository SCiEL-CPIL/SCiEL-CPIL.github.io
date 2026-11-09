"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with restart for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # For dim=5, lambda=10 is reasonable
        # We will use a fixed population size to simplify the loop and ensure deterministic call count
        # Standard CMA-ES uses lambda ~ 4 + floor(3 log n). For n=5, lambda=10.
        pop_size = 10
        
        # We need to manage the budget carefully.
        # Each generation consumes pop_size calls.
        # We will run generations until budget is exhausted or time limit is hit.
        
        best_f_global = np.inf
        best_x_global = None
        
        # Initialize mean in the center, as the problem space is [-5,5] and minimizer is in [-4,4]
        # Starting at 0 is a good heuristic for these types of benchmark functions.
        x_mean = np.zeros(dim)
        
        # Initial sigma
        # The domain is 10 units wide. A good initial step is around 20-30% of the domain size.
        sigma = 2.0
        
        # Covariance matrix C
        C = np.eye(dim)
        # Inverse square root of C (initially identity)
        C_inv_sqrt = np.eye(dim)
        # Eigen decomposition cache
        eigvals = np.ones(dim)
        eigvecs = np.eye(dim)
        
        # Adaptation parameters
        # Damping factors
        mu = pop_size // 2  # Weights for top mu individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # History for restart logic
        best_f_last_restart = np.inf
        stagnation_count = 0
        max_stagnation = 10 # Generations without improvement to trigger restart
        
        calls_made = 0
        
        # Main loop
        # We will run in generations. 
        # If we don't improve for 'max_stagnation' generations, we might restart.
        # However, CMA-ES is quite robust. A single run might be enough if sigma adapts well.
        # Given the budget of 500 and pop_size 10, we have 50 generations.
        # 50 generations is often enough for dim=5 CMA-ES to converge.
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            offspring = []
            for i in range(pop_size):
                # Standard normal
                z = rng.standard_normal(dim)
                # Transform using C^{-1/2}
                # y = C^{-1/2} z
                y = C_inv_sqrt @ z
                # x = x_mean + sigma * y
                x_new = x_mean + sigma * y
                # Clip to box
                x_new = np.clip(x_new, -5, 5)
                
                # Evaluate
                f_new = func(x_new)
                calls_made += 1
                offspring.append((x_new, f_new))
                
                if calls_made >= budget:
                    break
            
            if len(offspring) < pop_size:
                # We ran out of budget in the middle of a generation.
                # We can still evaluate the rest if we have budget left? No, we broke the loop.
                # But we might have evaluated some.
                # For CMA-ES, we need the full population to update statistics correctly.
                # If we have partial population, we can just stop here.
                # Or we can evaluate the rest if we have budget.
                # The loop condition `calls_made < budget` will be false next iteration.
                # Let's just break if we can't complete a generation.
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda x: x[1])
            
            # Update global best
            if offspring[0][1] < best_f_global:
                best_f_global = offspring[0][1]
                best_x_global = offspring[0][0].copy()
            
            # Update stagnation
            if offspring[0][1] < best_f_last_restart - 1e-8:
                best_f_last_restart = offspring[0][1]
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Check for restart
            if stagnation_count > max_stagnation:
                # Restart with a new random center
                # But we have limited budget. 
                # If we restart, we reset CMA-ES state.
                # Let's just break if we are stagnating significantly, as CMA-ES might be stuck in a local minimum (Lunacek has two).
                # Actually, CMA-ES is good at escaping local minima if sigma doesn't shrink too fast.
                # If it stagnates, it's likely converged to a local min.
                # A restart with a different center might find the global one.
                # Let's try a restart if we have budget left.
                if calls_made < budget - pop_size * 2: # Need enough budget for a few more generations
                    # Reset CMA-ES state
                    x_mean = rng.uniform(-4, 4, dim)
                    sigma = 2.0
                    C = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)
                    eigvals = np.ones(dim)
                    eigvecs = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    best_f_last_restart = np.inf
                    stagnation_count = 0
                    continue
                else:
                    break

            # Select best mu individuals
            selected = [x for x, f in offspring[:mu]]
            
            # Update mean
            x_new_mean = np.sum([w * x for w, x in zip(weights, selected)], axis=0)
            
            # Update sigma and covariance
            # Step size control
            y_vec = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_vec)
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // pop_size))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec
            
            # Update covariance
            delta_h = (h_sigma - 1) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C \
                + c_1 * np.outer(p_c, p_c) \
                + c_mu * np.sum([w * np.outer(y_i, y_i) for w, y_i in zip(weights, [ (x - x_mean) / sigma for x in selected])], axis=0)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            C += 1e-12 * np.eye(dim)
            
            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update mean
            x_mean = x_new_mean
            
            # Update C^{-1/2} via eigen decomposition
            # This is expensive but dim=5 is small.
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip eigenvalues
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                # Fallback to identity if decomposition fails
                C_inv_sqrt = np.eye(dim)
                C = np.eye(dim)

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
