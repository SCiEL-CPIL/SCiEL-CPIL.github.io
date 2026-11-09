import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = None
        best_f = np.inf
        
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                raise Exception("Budget or time limit exceeded")
            x = np.asarray(x)
            if x.shape != (dim,):
                x = x.reshape(dim)
            # Clip to bounds
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            fx = func(x)
            if np.isnan(fx):
                fx = 1e10
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Initialization with Halton sequence for better spread
        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q = q / base
                r = r + q * (index % base)
                index = index // base
            return r

        points = []
        # Use a smaller initial set to save budget for the main search
        n_init = 15
        for i in range(1, n_init + 1):
            p = np.array([halton_sequence(i, base) for base in range(2, 2 + dim)])
            p = p * 10.0 - 5.0
            points.append(p)
        
        # Add origin and a few structured points
        points.append(np.zeros(dim))
        points.append(np.ones(dim) * 2.0)
        points.append(np.ones(dim) * -2.0)

        # Evaluate initial points
        f_values = []
        for p in points:
            if calls >= budget or time.time() > deadline - 10:
                break
            try:
                f_val = wrapped_func(p)
                f_values.append(f_val)
            except Exception:
                break

        if best_x is None:
            # Fallback if no evaluation succeeded (unlikely)
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        # Sort initial points by function value
        valid_points = [(p, f) for p, f in zip(points, f_values) if not np.isnan(f)]
        valid_points.sort(key=lambda x: x[1])
        
        # Take the best initial point as the starting point for the main search
        x = valid_points[0][0].copy()
        fx = valid_points[0][1]
        
        # 2. Adaptive Coordinate Descent with Curvature Estimation
        # We use a 1-eval-per-dim approach to maximize exploration within the budget.
        # For each dimension, we maintain a step size and direction.
        # If a step is successful, we grow the step size (exploration).
        # If unsuccessful, we shrink and flip (exploitation/backtracking).
        
        directions = np.ones(dim)
        steps = np.full(dim, 2.0) # Start with moderate steps
        min_step = 1e-6
        max_step = 10.0
        
        # We will run this until budget or time is exhausted
        # The "curvature" idea is implicitly handled by the adaptive step size:
        # along the ridge (valley floor), steps will be large because they are consistently successful.
        # off the ridge, steps will shrink quickly.
        
        # To improve on simple coordinate descent, we can occasionally try a "jump" 
        # or use the history to estimate a better direction, but given the constraints,
        # robust adaptive step size is key.
        
        # Let's implement a simple adaptive coordinate descent loop.
        # We cycle through dimensions.
        
        i = 0
        while calls < budget and time.time() < deadline:
            d = i % dim
            
            # Try a step in dimension d
            y = x.copy()
            y[d] = np.clip(y[d] + directions[d] * steps[d], -5.0, 5.0)
            
            try:
                fy = wrapped_func(y)
            except Exception:
                break
            
            if fy < fx:
                # Success
                x = y
                fx = fy
                # Grow step size
                steps[d] = min(steps[d] * 1.2, max_step)
                # Keep direction
            else:
                # Failure
                # Shrink step size
                steps[d] = max(steps[d] * 0.7, min_step)
                # Flip direction
                directions[d] *= -1
                
            i += 1
            
            # Periodically, we might want to reset directions or steps if we are stuck?
            # The flip mechanism handles local minima to some extent.
            # If we are stuck in a flat region, steps will grow, which is good for escaping.
            
            # To add a bit of "curvature" awareness, we can check if the step in dimension d
            # was significantly better than the previous step in the same dimension?
            # This is complex to track. Let's stick to the simple adaptive rule which is robust.
            
            # One improvement: if steps[d] becomes very small, it might be worth trying
            # a larger random step in that dimension to escape a local trap.
            # But the flip+grow mechanism should handle this.

        return best_x, best_f
