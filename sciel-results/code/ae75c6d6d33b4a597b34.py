import numpy as np
import time

class Optimizer:
    """Deterministic lattice search followed by adaptive (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        n = self.dim
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            if calls >= budget:
                return np.inf
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Deterministic lattice search on a 3^5 grid
        # This gives 243 evaluations covering the entire [-5, 5]^5 space
        grid = np.linspace(-4.5, 4.5, 3)
        points = list(np.meshgrid(*[grid] * n, indexing='ij'))
        points = [np.column_stack(pts).reshape(-1, n) for pts in [np.meshgrid(*[grid] * n, indexing='ij')]]
        all_points = points[0]
        
        for x in all_points:
            if calls >= budget or time.time() > deadline:
                break
            evaluate(x)

        if best_x is None:
            return (np.zeros(dim), 0.0)

        # Phase 2: (1+1)-ES refinement
        # Weierstrass function has fine-scale structure; start with moderate step size
        sigma = 0.5
        x = best_x.copy()
        f = best_f
        
        # Success rate tracking for adaptive step size
        success_count = 0
        total_count = 0
        window_size = 10
        
        while calls < budget and time.time() < deadline:
            # Generate candidate
            direction = np.random.randn(dim)
            direction /= np.linalg.norm(direction)
            x_trial = x + sigma * direction
            
            # Clip to bounds
            x_trial = np.clip(x_trial, -5.0, 5.0)
            
            f_trial = evaluate(x_trial)
            if f_trial == np.inf:
                break
                
            total_count += 1
            if f_trial < f:
                x = x_trial
                f = f_trial
                success_count += 1
                
                # Update global best
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            else:
                # Reject
                pass
            
            # Adapt sigma based on success rate over recent window
            if total_count % window_size == 0:
                rate = success_count / total_count
                # Target success rate ~1/3
                if rate > 0.4:
                    sigma *= 1.2
                elif rate < 0.1:
                    sigma *= 0.8
                success_count = 0
                total_count = 0

        return best_x, best_f
