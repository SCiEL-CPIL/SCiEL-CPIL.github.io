"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with 1/5 success rule for BBOB-style Schwefel function on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # (1+1)-ES with 1/5 success rule
        # Initial step size: cover the search space
        sigma = 3.0
        
        # Start from origin
        x = np.zeros(dim)
        
        # Track best
        best_x = x.copy()
        best_f = np.inf
        
        try:
            f_current = func(x)
        except Exception:
            f_current = np.inf
        
        evals = 1
        
        if f_current < best_f:
            best_f = f_current
            best_x = x.copy()
            
        # 1/5 rule parameters
        # We adapt sigma based on the success rate over a window of generations
        # Standard 1/5 rule: if success rate > 1/5, increase sigma; if < 1/5, decrease.
        # To be more stable, we can use a moving average or just simple multiplicative updates.
        # Let's use a simple multiplicative update based on immediate success, 
        # or a windowed approach. 
        # For speed and simplicity, we'll use a window of 10 generations.
        
        window_size = 10
        successes = 0
        window_count = 0
        
        # Main loop
        while evals < budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            z = rng.standard_normal(dim)
            x_offspring = x + sigma * z
            np.clip(x_offspring, -5, 5, out=x_offspring)
            
            try:
                f_offspring = func(x_offspring)
            except Exception:
                f_offspring = np.inf
                
            evals += 1
            
            # Update best
            if f_offspring < best_f:
                best_f = f_offspring
                best_x = x_offspring.copy()
                
            # Update current position and success count
            if f_offspring < f_current:
                successes += 1
                x = x_offspring
                f_current = f_offspring
            else:
                # If offspring is worse, we stay at current x.
                # Note: In strict (1+1)-ES, we usually accept if better, else reject.
                # The 1/5 rule counts successes.
                pass
                
            window_count += 1
            
            # Adapt step size every window_size generations
            if window_count >= window_size:
                success_rate = successes / window_count
                # 1/5 rule: target success rate is 0.2
                # If success rate > 0.2, increase sigma (move was too small)
                # If success rate < 0.2, decrease sigma (move was too large)
                
                # Simple multiplicative adaptation
                # sigma *= (1 + c * (success_rate - 0.2))
                # To avoid drastic changes, we can use:
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 0.9
                else:
                    pass # No change
                
                # Clip sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                # Reset window
                successes = 0
                window_count = 0
        
        return best_x, best_f
# EVOLVE-BLOCK-END
