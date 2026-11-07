import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 290.0
        
        budget = self.budget
        dim = self.dim
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            try:
                val = func(x)
            except Exception:
                return None
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val
        
        # (1+1)-ES with adaptive step size
        # For BBOB f13 (Sharp Ridge), a (1+1)-ES with specific sigma adaptation
        # can perform well. We use a simple strategy:
        # - Start with a moderate sigma
        # - Increase sigma on success, decrease on failure
        # - This allows for both exploration and exploitation
        
        # Initial point: sample in [-4, 4]
        x = rng.uniform(-4, 4, dim)
        fx = evaluate(x)
        if fx is None:
            return best_x, best_f
        
        # Adaptive step size
        sigma = 1.0
        success_threshold = 50  # Number of successes before increasing sigma
        failures = 0
        successes = 0
        
        while calls < budget and time.time() < deadline:
            # Generate a perturbation
            direction = rng.standard_normal(dim)
            x_trial = x + sigma * direction
            
            # Clip to bounds (though func also clips, this is safer for our tracking)
            x_trial = np.clip(x_trial, -5, 5)
            
            fx_trial = evaluate(x_trial)
            if fx_trial is None:
                break
            
            # Update based on success/failure
            if fx_trial < fx:
                # Success: move to new point
                x = x_trial
                fx = fx_trial
                successes += 1
                failures = 0
                
                # Adapt sigma: increase slightly on success
                # The inspiration suggests sigma *= 1.5 on success
                # But 1.5 might be too aggressive. Let's try 1.2 for stability
                # Actually, let's stick to the inspiration's finding: 1.5
                sigma *= 1.5
            else:
                # Failure: stay at current point
                failures += 1
                successes = 0
                
                # Adapt sigma: decrease on failure
                # The inspiration suggests sigma *= 0.87 on failure
                sigma *= 0.87
            
            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-4, 10.0)
            
            # If we've had many consecutive successes, we might be in a flat region
            # or approaching the minimum. We can reduce the increase rate.
            # For now, keep it simple.
            
            # Optional: If sigma is very small and we're not improving, 
            # we could restart from a random point. But with 500 budget,
            # let's just continue.
        
        return best_x, best_f
