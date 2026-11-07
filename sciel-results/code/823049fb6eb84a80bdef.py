import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard: 300s limit, keep 10s margin
        deadline = time.time() + 290
        
        # Phase 1: Multi-start exploration
        # Evaluate a small number of random points to find a good starting basin
        n_starts = 10
        best_x = None
        best_f = float('inf')
        
        # Use a fixed seed for reproducibility in the "seed" context, 
        # but the task says "hidden seeded shift", implying the func is deterministic per instance.
        # We should use a random seed for our internal RNG to avoid biasing towards specific instances if the evaluator re-runs with different hidden seeds.
        # However, using a fixed seed for our *internal* randomness is fine as long as it's not correlated with the hidden shift in a bad way.
        # Standard practice: use np.random.default_rng() without seed or with a time-based seed?
        # The task says "Instances apply a hidden seeded shift". This means func(x) is deterministic.
        # Our internal randomness should be random to explore.
        rng = np.random.default_rng()
        
        for _ in range(n_starts):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x
        
        x = best_x
        fx = best_f
        calls_used = n_starts
        
        # Phase 2: (1+1)-ES Optimization
        # Initial sigma: scale with dimension? Or just 1.0?
        # Domain is [-5, 5], so width 10. 1.0 is 10% of the range.
        sigma = 1.0
        
        # Adaptation factors
        # For (1+1)-ES, 1/5 success rule suggests sigma *= 1.5 on success, 0.85 on failure.
        # Let's use slightly more conservative growth to avoid oscillating.
        sigma_inc = 1.2
        sigma_dec = 0.8
        
        while calls_used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_used += 1
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= sigma_inc
            else:
                sigma *= sigma_dec
            
            # Optional: If sigma becomes very small, we might be stuck.
            # But with 500 budget, we likely won't reach sigma < 1e-6 unless we are very close.
            
        return x, fx
