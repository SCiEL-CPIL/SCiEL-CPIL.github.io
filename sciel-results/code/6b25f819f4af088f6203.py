import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        calls_used = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = None
        dim = self.dim

        def update_best(x, f):
            nonlocal best_f, best_x
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = []
            for i in range(1, n + 1):
                point = [halton(i, base) for base in range(2, dim + 2)]
                # Map [0,1] to [-5,5]
                point = [p * 10 - 5 for p in point]
                points.append(np.array(point))
            return points

        # Phase 1: Global Exploration with Halton Sequence
        # Use ~100 evaluations for global coverage
        num_halton = min(100, max_calls // 5)
        halton_points = halton_sequence(num_halton, dim)
        
        for x in halton_points:
            if calls_used >= max_calls or time.time() > deadline:
                break
            f_val = func(x)
            calls_used += 1
            update_best(x, f_val)

        # Phase 2: Multi-start L-BFGS-B Local Refinement
        # Use remaining budget for local optimization
        # We'll start from the best points found in Phase 1 and a few random points
        # L-BFGS-B is efficient for smooth functions and handles bounds well
        
        # Create a list of starting points for local optimization
        # We'll use the best few from Halton and some random points
        # Since we don't store all Halton points with their values, we'll just use the best_x found so far
        # and generate a few random starts
        
        # Actually, let's collect the best few points from the Halton phase
        # But we didn't store them. Let's just use best_x and generate new random starts.
        
        # Better approach: Use a fixed number of local optimizations
        # Each L-BFGS-B call uses multiple function evaluations
        # We need to be careful with the budget
        
        # Let's define a wrapper that tracks calls and stops if budget exceeded
        class CallCounter:
            def __init__(self, func, max_calls, deadline):
                self.func = func
                self.calls = 0
                self.max_calls = max_calls
                self.deadline = deadline
                self.best_f = float('inf')
                self.best_x = None
                self.stopped = False

            def __call__(self, x):
                if self.stopped:
                    return self.best_f
                if time.time() > self.deadline or self.calls >= self.max_calls:
                    self.stopped = True
                    return self.best_f
                f_val = self.func(x)
                self.calls += 1
                if f_val < self.best_f:
                    self.best_f = f_val
                    self.best_x = x.copy()
                return f_val

        # We'll run L-BFGS-B from a few starting points
        # Starting points: best_x from Phase 1, and a few random points
        start_points = [best_x]
        
        # Add a few random points
        rng = np.random.default_rng(42)
        for _ in range(5):
            x_rand = rng.uniform(-5, 5, size=dim)
            start_points.append(x_rand)

        # Run L-BFGS-B from each start point
        for x0 in start_points:
            if calls_used >= max_calls or time.time() > deadline:
                break
            
            # Create a counter for this specific optimization
            counter = CallCounter(func, max_calls - calls_used, deadline)
            
            try:
                result = minimize(
                    counter, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfun': 50}  # Limit function calls
                )
                
                # Update global best
                if counter.best_x is not None:
                    update_best(counter.best_x, counter.best_f)
                
                calls_used += counter.calls
                
            except Exception:
                # If L-BFGS-B fails, just continue
                pass

        # Phase 3: Final random sampling if budget remains
        while calls_used < max_calls and time.time() < deadline:
            x_rand = rng.uniform(-5, 5, size=dim)
            f_rand = func(x_rand)
            calls_used += 1
            update_best(x_rand, f_rand)

        return best_x, best_f
