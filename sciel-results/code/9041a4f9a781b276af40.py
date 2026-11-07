import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s kill
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Global best tracking
        best_x = np.zeros(n)
        best_f = np.inf
        calls_made = 0

        # Parameters for (1+8)-ES
        # Using a small population allows for faster convergence and better ridge exploration
        # without the overhead of CMA-ES covariance updates.
        pop_size = 8
        # Initial step size. Box is [-5, 5], so 2.0 covers a good portion of the space.
        sigma = 2.0
        
        # Success rate target for step size adaptation
        # We want to maintain a success rate of ~20% for efficient exploration
        target_success_rate = 0.2
        
        # Stagnation detection
        # If we don't improve the global best for 'stagnation_limit' generations, we restart
        stagnation_limit = 10
        no_improvement_count = 0
        
        # Initial evaluation at center
        x0 = np.zeros(n)
        f0 = func(x0)
        calls_made += 1
        best_x = x0.copy()
        best_f = f0

        while calls_made < budget and time.time() < deadline:
            # Generate a population of candidates around the current best
            # We use the global best as the center for the next generation
            # This is a "re-starting" strategy implicitly: if we stagnate, we reset sigma
            # and continue from the best.
            
            candidates = np.zeros((pop_size, n))
            for i in range(pop_size):
                z = rng.standard_normal(n)
                candidates[i] = best_x + sigma * z
                candidates[i] = np.clip(candidates[i], -5, 5)
            
            # Evaluate all candidates
            fs = np.array([func(y) for y in candidates])
            calls_made += pop_size
            
            # Find the best candidate in this generation
            idx_best = np.argmin(fs)
            f_best_gen = fs[idx_best]
            x_best_gen = candidates[idx_best].copy()
            
            # Update global best
            if f_best_gen < best_f:
                best_f = f_best_gen
                best_x = x_best_gen
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Adapt step size based on success rate
            # If we found a better point than the current best, it's a success.
            # Actually, a better metric for (1+lambda) is: did any child beat the parent (current best)?
            # Here, 'parent' is best_x from previous iteration.
            # If best_f improved, success = True.
            # To be more robust, we can count how many children were better than the parent.
            # But standard (1+lambda) adaptation often just checks if the best child was better.
            # Let's use a slightly more sophisticated adaptation:
            # If we improved, increase sigma slightly to explore more.
            # If we didn't improve, decrease sigma to converge.
            
            # However, to handle the "sharp ridge", we want to maintain some exploration.
            # If no_improvement_count is high, we might be stuck.
            
            if f_best_gen < best_f:
                # Improved
                sigma *= 1.05 # Slight expansion
            else:
                # No improvement
                sigma *= 0.95 # Slight contraction
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-4, 5.0)
            
            # Stagnation handling
            if no_improvement_count >= stagnation_limit:
                # Reset sigma to a moderate value to escape potential local flat areas
                # and restart exploration from the global best
                sigma = 1.5
                no_improvement_count = 0
                # Optional: small random perturbation of best_x to break symmetry?
                # No, best_x is the best we know. Just changing sigma is usually enough.
                
            # Check time and budget
            if calls_made >= budget or time.time() >= deadline:
                break

        # Use remaining budget for single step (1+1) if available, with very small steps for final polish
        if calls_made < budget and time.time() < deadline:
            sigma_11 = sigma * 0.1
            if sigma_11 < 1e-4:
                sigma_11 = 1e-4
            
            while calls_made < budget and time.time() < deadline:
                z = rng.standard_normal(n)
                y = best_x + sigma_11 * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    # Keep sigma_11 constant or slightly increase if we are still improving?
                    # For final polish, keep it small and stable.
                else:
                    # If we are in the final polish phase, we don't adapt as aggressively
                    pass

        return best_x, best_f
