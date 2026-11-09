import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin

        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        # 1. Initialization: Halton sequence + structured points
        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q = q / base
                r = r + q * (index % base)
                index = index // base
            return r

        points = []
        n_init = 15
        for i in range(1, n_init + 1):
            p = np.array([halton_sequence(i, base) for base in range(2, 2 + dim)])
            p = p * 10.0 - 5.0
            points.append(p)
        
        # Add origin and some structured points to ensure coverage near center
        points.append(np.zeros(dim))
        points.append(np.ones(dim) * 2.0)
        points.append(np.ones(dim) * -2.0)

        # Evaluate initial points
        f_values = []
        for p in points:
            if calls >= budget or time.time() > deadline - 5:
                break
            try:
                f_val = func(p)
            except:
                f_val = np.inf
            calls += 1
            f_values.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = p.copy()

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # 2. Adaptive (1+1)-ES with 1/5 rule
        # This is robust for ridge functions and adapts step size to the landscape.
        # It allows for significant progress in the first few dozen calls.
        
        # Start step size: roughly 1/5th of the box width
        sigma = 1.0
        # 1/5 rule: if success rate is > 0.2, increase sigma; if < 0.2, decrease
        # Window size for success rate calculation
        window_size = 15
        successes = 0
        total_evals_in_window = 0
        
        # We will run the ES until budget or time is exhausted.
        # To avoid getting stuck in a local minimum, we will occasionally restart
        # or perturb if no improvement is seen for a while, but for a single
        # minimizer problem, a well-tuned (1+1)-ES is often sufficient.
        
        # However, to be safe against local minima, we can keep a small population
        # or just rely on the initial spread. Given the budget is small (500),
        # a focused search from the best initial point is likely best.
        
        # Let's refine the (1+1)-ES parameters.
        # Standard 1/5 rule:
        # sigma_new = sigma * exp(0.33 * (success_rate - 0.2) / 0.2) ?
        # Or simpler:
        # if success_rate > 0.2: sigma *= 1.05
        # else: sigma /= 1.05
        
        # Let's use a slightly more aggressive adaptation to find the ridge quickly.
        
        for _ in range(budget - calls):
            if time.time() > deadline - 5:
                break
            
            # Generate a candidate
            # Use standard normal noise
            noise = np.random.randn(dim)
            x_new = best_x + sigma * noise
            
            # Clip to bounds
            x_new = np.clip(x_new, -5.0, 5.0)
            
            try:
                f_new = func(x_new)
            except:
                f_new = np.inf
            
            calls += 1
            
            # Update success rate
            success = f_new < best_f
            if success:
                successes += 1
                best_f = f_new
                best_x = x_new.copy()
            
            total_evals_in_window += 1
            
            # Adapt sigma every window_size steps
            if total_evals_in_window >= window_size:
                success_rate = successes / total_evals_in_window
                # 1/5 rule: target success rate of 0.2
                # If success rate is too high, we are moving too slowly (or noise is too small)
                # If success rate is too low, we are moving too fast (or noise is too large)
                
                # Multiplicative update
                if success_rate > 0.2:
                    sigma *= 1.05
                else:
                    sigma /= 1.05
                
                # Reset window
                successes = 0
                total_evals_in_window = 0
                
                # Safety bounds for sigma
                if sigma < 1e-6:
                    sigma = 1e-6
                if sigma > 10.0:
                    sigma = 10.0

        return best_x, best_f
