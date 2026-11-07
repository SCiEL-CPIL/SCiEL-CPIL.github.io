import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        # Best solution tracking
        best_x = None
        best_f = np.inf
        calls = 0
        
        def time_ok():
            return time.time() < deadline
            
        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # 1. Initial coarse scan to find a good starting region
        # Use a grid of points or random sampling to identify the basin
        # Since dim=5, a small grid or random set is cheap.
        # Let's try a few random points and the origin.
        init_points = [np.zeros(dim)]
        # Add some random points to avoid getting stuck in a bad local minimum
        for _ in range(10):
            init_points.append(np.random.uniform(-5, 5, dim))
        
        # Evaluate initial points
        for x in init_points:
            if not time_ok() or calls >= budget: break
            f = func(x)
            calls += 1
            record(x, f)
        
        if best_x is None:
             return None, np.inf

        # 2. Run (1+1)-ES with adaptive step size
        # This is a structural change from CMA-ES population to single-step ES
        # (1+1)-ES is often faster for low-dimensional, smooth functions if tuned well
        x = best_x.copy()
        f_current = best_f
        sigma = 1.0 # Start with a moderate step size
        
        # Success rate control parameters
        target_success = 0.2 # Aim for 20% success rate to balance exploration/exploitation
        success_count = 0
        total_generations = 0
        window_size = 10
        
        while calls < budget and time_ok():
            # Generate a random direction
            z = np.random.randn(dim)
            z /= np.linalg.norm(z)
            
            # Propose new point
            x_new = x + sigma * z
            # Clip to box (though func clips anyway, keeping it in bounds helps ES)
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate
            if not time_ok() or calls >= budget: break
            f_new = func(x_new)
            calls += 1
            record(x_new, f_new)
            
            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
            
            # Adapt sigma based on success
            success = (f_new < f_current)
            success_count += success
            total_generations += 1
            
            # Check success rate over the last window
            if total_generations >= window_size:
                current_success_rate = success_count / window_size
                # Adjust sigma
                if current_success_rate > target_success:
                    sigma *= 1.5 # Increase step size
                elif current_success_rate < target_success:
                    sigma /= 1.5 # Decrease step size
                
                # Reset window
                success_count = 0
                total_generations = 0
            
            # Accept the new point if it's better (1+1 rule)
            if f_new < f_current:
                x = x_new
                f_current = f_new
            # If not better, we keep x and f_current, but we've already updated best_x if it was global best
            # In (1+1), if we reject, we stay at the current point.
            
            # Convergence check
            if sigma < 1e-8:
                break

        return best_x, best_f
