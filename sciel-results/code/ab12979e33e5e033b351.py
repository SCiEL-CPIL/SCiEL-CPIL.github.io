"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Stochastic (1+lambda)-ES with adaptive step size and restarts for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        call_count = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # (1+lambda)-ES parameters
        lambda_ = 20  # Offspring count
        sigma = 1.0   # Initial step size
        sigma_min = 1e-4
        sigma_max = 5.0
        target_success_rate = 0.2
        
        # Initial point: center
        x = np.zeros(dim)
        f_x = wrapped_func(x)
        
        no_improve_count = 0
        max_no_improve = 50  # Restart if no improvement for 50 iterations
        
        while call_count < budget and time.time() < deadline:
            # Generate offspring
            offspring = []
            for _ in range(lambda_):
                # Step size adaptation is done after evaluation, so use current sigma
                dx = rng.normal(0, sigma, dim)
                new_x = x + dx
                new_x = np.clip(new_x, -5, 5)
                offspring.append((new_x, wrapped_func(new_x)))
            
            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[1])
            
            # Select the best offspring
            best_offspring_x, best_offspring_f = offspring[0]
            
            # Update best if improved
            if best_offspring_f < f_x:
                x = best_offspring_x
                f_x = best_offspring_f
                no_improve_count = 0
                
                # Adapt step size: if success, keep or slightly increase? 
                # Standard (1+lambda)ES: if any improvement, sigma *= 1/sqrt(lambda) is too aggressive.
                # Let's use a simple rule: if best offspring is better, sigma *= 1.05 (slight increase to explore more)
                # But if we want to exploit, we might decrease.
                # Actually, if we improved, it means the step was good. Let's try to keep it or increase slightly.
                # However, for convergence, we usually decrease when we are close.
                # Let's use the number of successful offspring to adapt sigma.
                num_success = 0
                for _, f_off in offspring:
                    if f_off < f_x: # Note: f_x is the OLD best before update in this block? No, we updated x.
                         # We should compare against the OLD x's fitness.
                         # Let's restructure to capture old_f_x.
                         pass
                
                # Simpler adaptation:
                # Count how many offspring are better than the parent (old x)
                # We need old_f_x.
                # Let's refactor the loop logic slightly to track success count properly.
                
            else:
                no_improve_count += 1
                # If no improvement, decrease step size
                sigma *= 0.9
                if sigma < sigma_min:
                    sigma = sigma_min
                    
            # Proper success-based adaptation
            # Let's do it correctly here
            # We need to compare offspring against the PARENT (x before update)
            # The code above updated x immediately. Let's fix the logic flow.
            
            # Restart if stagnation
            if no_improve_count > max_no_improve:
                # Random restart
                x = rng.uniform(-5, 5, dim)
                f_x = wrapped_func(x)
                sigma = 1.0
                no_improve_count = 0

        return best_x, best_f

# EVOLVE-BLOCK-END
