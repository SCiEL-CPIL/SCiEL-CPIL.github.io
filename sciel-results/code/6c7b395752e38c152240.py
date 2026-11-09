"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid Multi-Start (1+1)-ES Optimizer.
    
    Strategy:
    1. Initial Diversification: Sample diverse points (origin, random, mid-range) 
       to identify a promising basin. This is crucial for multimodal functions 
       like Rastrigin and Gallagher where the global minimum is not near the origin.
    2. Exploitation: Run an aggressive (1+1)-Evolution Strategy from the best 
       initial point. This algorithm has minimal overhead per iteration, allowing 
       for rapid step-size adaptation and frequent updates, which is highly 
       beneficial for the AOCC score that rewards early progress.
    3. Safety: Strict deadline and budget enforcement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. We stop at 290s to be safe.
        deadline = start_time + 290.0 
        
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        calls_made = 0

        best_x = None
        best_f = np.inf

        # Phase 1: Initial Diversification
        # We use ~10-15% of the budget for exploration to ensure we don't get stuck 
        # in a poor local minimum, which is common in Rastrigin/Gallagher.
        n_init_samples = 10
        init_candidates = []
        
        # 1. Origin
        init_candidates.append(np.zeros(n))
        
        # 2. Mid-range points (helps if the shift is moderate)
        init_candidates.append(np.ones(n) * 2.0)
        init_candidates.append(np.ones(n) * -2.0)
        init_candidates.append(np.ones(n) * 1.0)
        init_candidates.append(np.ones(n) * -1.0)
        
        # 3. Random points
        for _ in range(n_init_samples - 5):
            init_candidates.append(rng.uniform(-5, 5, n))

        for x in init_candidates:
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made = 1

        # Phase 2: Aggressive (1+1)-ES from the best initial point
        x = best_x.copy()
        fx = best_f
        
        # Initial step size: 2.0 is aggressive enough to escape local minima 
        # in 5D within the remaining budget, but small enough to converge.
        sigma = 2.0
        
        # Parameters for adaptation
        sigma_inc = 1.5
        sigma_dec = 0.87
        sigma_min = 1e-4
        sigma_max = 10.0

        remaining_budget = total_budget - calls_made

        # Main Optimization Loop
        for _ in range(remaining_budget):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            # Generate offspring
            # Standard normal noise
            noise = rng.standard_normal(n)
            y = x + sigma * noise
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Success: move to y, increase step size
                x, fx = y, fy
                sigma *= sigma_inc
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Failure: stay at x, decrease step size
                sigma *= sigma_dec

            # Clamp sigma to valid range
            sigma = max(sigma_min, min(sigma, sigma_max))
            
            # Optional: If sigma becomes extremely small, we might be stuck.
            # However, with the diversification phase, we are likely in a good basin.
            # We do not restart to preserve the continuous history of best_x for AOCC.

        return best_x, best_f
# EVOLVE-BLOCK-END
