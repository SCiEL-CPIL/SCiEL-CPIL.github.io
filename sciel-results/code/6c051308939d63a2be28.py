import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s kill
        
        # State tracking
        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0
        rng = np.random.default_rng(42)
        
        # (1+1)-ES Parameters
        # Start at origin.
        x_current = np.zeros(dim)
        f_current = func(x_current)
        calls += 1
        best_x = x_current.copy()
        best_f = f_current
        
        # Initial step size.
        # Domain [-5, 5]. Distance from origin to corner is ~11.18.
        # Start with a step size that allows exploration of the box.
        sigma = 1.5
        
        # 1/5th rule adaptation
        # Window size for success rate calculation
        window_size = 15
        successes_window = []
        
        # Minimum and maximum step sizes to prevent divergence or stagnation
        sigma_min = 1e-8
        sigma_max = 10.0
        
        while calls < budget and time.time() < deadline:
            # Generate mutation
            z = rng.standard_normal(dim)
            x_trial = x_current + sigma * z
            
            # Clip to bounds
            x_trial = np.clip(x_trial, -5.0, 5.0)
            
            # Evaluate
            f_trial = func(x_trial)
            calls += 1
            
            # Determine success
            # Success if trial is better than current
            success = (f_trial < f_current)
            
            # Update global best
            if f_trial < best_f:
                best_f = f_trial
                best_x = x_trial.copy()
                
            # Update ES state
            if success:
                x_current = x_trial
                f_current = f_trial
                successes_window.append(1)
            else:
                successes_window.append(0)
                
            # Adaptation
            # Only adapt if we have enough history
            if len(successes_window) >= window_size:
                # Keep only the last 'window_size' entries
                if len(successes_window) > window_size:
                    successes_window.pop(0)
                    
                success_rate = sum(successes_window) / window_size
                
                # 1/5th rule: sigma = sigma * exp((sr - 0.2) / 5)
                # This is standard. For faster convergence on Sphere, we might want a steeper curve.
                # However, 1/5th is robust. Let's stick to it but maybe adjust the exponent.
                # Standard is /5. Let's try /3 for faster adaptation?
                # Or just stick to standard.
                factor = np.exp((success_rate - 0.2) / 5.0)
                sigma *= factor
                
                # Clamp sigma
                if sigma < sigma_min:
                    sigma = sigma_min
                elif sigma > sigma_max:
                    sigma = sigma_max
                    
                # If we have converged to a very small error, we might want to stop 
                # or just let it run out of budget.
                # AOCC rewards early progress.
                # If best_f is extremely close to 0 (or offset), we are done.
                # But we don't know offset. 
                # If sigma is very small, we are stuck.
                # The 1/5th rule handles this: if sr < 0.2, sigma decreases.
                # If sr > 0.2, sigma increases.
                # On Sphere, as we get closer, the optimal sr drops below 0.2? 
                # Actually, for Sphere, the optimal step size decreases.
                # The 1/5th rule adapts sigma to keep sr ~ 0.2.
                # This is effective.

        return best_x, best_f
