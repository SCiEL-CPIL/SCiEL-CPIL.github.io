"""Model-and-optimize approach for BBOB-style Schwefel f20 on [-5,5]^5."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    """Multi-restart (1+1)-ES for BBOB-style Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_exceeded():
            return time.time() > deadline

        best_f = np.inf
        best_x = np.zeros(dim)
        evals = 0

        # Multi-restart (1+1)-ES
        # We will use the entire budget for this process.
        # Strategy: 
        # 1. Initialize x at a random point or 0.
        # 2. Use a step size sigma.
        # 3. Generate mutant y = x + sigma * N(0, I).
        # 4. If f(y) < f(x), accept y, increase sigma slightly (or keep).
        # 5. If f(y) >= f(x), reject, decrease sigma.
        # 6. If no improvement for 'stagnation_limit' steps, restart with new x0.
        
        # Parameters
        # In 5D, a good initial sigma is around 1.0-2.0. 
        # Given the box is [-5, 5], let's start with sigma=2.0.
        # Stagnation limit: 20-30 evaluations without improvement triggers a restart.
        
        stagnation_limit = 30
        max_sigma = 5.0
        min_sigma = 1e-3
        sigma = 2.0
        
        # We'll manage restarts manually to control the budget precisely.
        # Total budget is 500.
        
        while evals < budget and not time_exceeded():
            # Initialize new lineage
            # First restart at 0, others random
            if evals == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-5, 5, size=dim)
            
            try:
                f_x = func(x)
                evals += 1
            except Exception:
                f_x = np.inf
                evals += 1
                # If it fails, try to move on
                if evals >= budget: break

            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()
            
            sigma = 2.0 # Reset sigma for each restart to allow wide exploration again
            stagnation_count = 0

            # Run (1+1)-ES loop for this restart
            while evals < budget and not time_exceeded():
                # Generate mutant
                y = x + sigma * rng.standard_normal(dim)
                
                # Clip to box [-5, 5]
                y = np.clip(y, -5, 5)
                
                try:
                    f_y = func(y)
                    evals += 1
                except Exception:
                    f_y = np.inf
                    evals += 1
                    if evals >= budget: break

                if f_y < f_x:
                    # Accept
                    x = y
                    f_x = f_y
                    if f_x < best_f:
                        best_f = f_x
                        best_x = x.copy()
                    stagnation_count = 0
                    # Increase sigma slightly to keep exploring, but cap it
                    # A common rule is sigma *= 1.1 if success, /1.1 if fail
                    # But for Schwefel, we might want to keep sigma larger to escape local minima
                    # Let's use a more aggressive increase
                    sigma = min(sigma * 1.1, max_sigma)
                else:
                    # Reject
                    stagnation_count += 1
                    # Decrease sigma
                    sigma = max(sigma * 0.9, min_sigma)
                    
                # Check for stagnation to trigger restart
                if stagnation_count >= stagnation_limit:
                    break

        return best_x, best_f
