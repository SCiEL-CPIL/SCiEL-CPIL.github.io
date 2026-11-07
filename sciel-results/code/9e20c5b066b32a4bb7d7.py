import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, use 290s to be safe
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility, though the problem instances
        # have hidden shifts/rotations, the search strategy itself can be deterministic
        # relative to the function evaluations.
        rng = np.random.default_rng(42)
        
        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        calls_made = 1
        best_x = x.copy()
        best_f = fx
        
        # Initial step size: 
        # The box is [-5, 5]^5. A reasonable initial sigma is 2.0 to cover the space quickly.
        sigma = 2.0
        
        # (1+1)-ES parameters
        # 1/5th success rule: increase sigma if >20% of trials are successful, decrease if <20%
        # For (1+1), we do 1 trial per iteration. 
        # We need to accumulate successes over a window to apply the rule effectively.
        # Alternatively, we can use a simpler adaptive scheme: 
        # sigma *= exp(c_sigma * (successes_in_window / window_size - 0.2))
        # Let's use a window size of 10 iterations for stability.
        window_size = 10
        successes = 0
        window_count = 0
        c_sigma = 0.5 # Learning rate for sigma adaptation
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate trial point
            # Standard normal perturbation
            z = rng.standard_normal(dim)
            x_trial = x + sigma * z
            
            # Clip to box [-5, 5]
            x_trial = np.clip(x_trial, -5, 5)
            
            # Evaluate
            fx_trial = func(x_trial)
            calls_made += 1
            
            # Check for success
            is_success = fx_trial < fx
            
            # Track successes for adaptation
            successes += 1 if is_success else 0
            window_count += 1
            
            if is_success:
                x = x_trial
                fx = fx_trial
                best_x = x.copy()
                best_f = fx
            else:
                # Reject step, keep current x and fx
                pass
            
            # Adapt sigma every window_size steps
            if window_count >= window_size:
                success_rate = successes / window_count
                # 1/5th rule: target 0.2 success rate
                # If success rate > 0.2, increase sigma
                # If success rate < 0.2, decrease sigma
                # Log-linear update
                sigma *= np.exp(c_sigma * (success_rate - 0.2))
                
                # Clamp sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                successes = 0
                window_count = 0
        
        return best_x, best_f
