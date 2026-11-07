"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-restart L-BFGS-B optimizer exploiting smooth structure of f11."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        d = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        calls = 0
        best_x = np.zeros(d)
        best_fx = np.inf
        
        def safe_func(x):
            nonlocal calls, best_x, best_fx
            if calls >= budget:
                return best_fx
            x = np.clip(x, -5, 5)
            calls += 1
            fx = float(func(x))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Generate deterministic grid of initial points for multi-restart
        # Use a stratified sample to cover [-5, 5]^d efficiently
        n_restarts = 10
        rng = np.random.RandomState(42)
        
        # Create diverse starting points
        # Mix of center, corners, and random points
        start_points = []
        
        # Center
        start_points.append(np.zeros(d))
        
        # Corners (sample a few)
        for _ in range(4):
            p = rng.uniform(-5, 5, size=d)
            start_points.append(p)
        
        # Points along axes
        for i in range(d):
            p = np.zeros(d)
            p[i] = 3.0
            start_points.append(p)
        
        # Fill remaining with random points
        while len(start_points) < n_restarts:
            p = rng.uniform(-5, 5, size=d)
            start_points.append(p)
        
        # Run L-BFGS-B from each starting point
        bounds = [(-5, 5)] * d
        
        for x0 in start_points:
            if not time_ok() or calls >= budget - 5:
                break
            
            def local_func(v):
                nonlocal calls
                if calls >= budget:
                    return best_fx
                return safe_func(v)
            
            try:
                res = minimize(
                    local_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'maxls': 30, 'ftol': 1e-14, 'gtol': 1e-10, 'eps': 1e-8}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
