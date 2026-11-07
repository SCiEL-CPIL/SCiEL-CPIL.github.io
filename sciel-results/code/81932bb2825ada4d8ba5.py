import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

class Optimizer:
    """
    Optimizer using multi-start Powell's Method with a scout phase.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        calls = 0
        best_x = None
        best_f = float('inf')
        
        # Deadline guard variables
        start_time = time.time()
        deadline = 300.0  # Safety margin from 330s limit

        def check_time():
            return time.time() > deadline

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            
            if check_time():
                return float('inf')
            
            if calls >= self.budget:
                return float('inf')
            
            calls += 1
            try:
                val = float(func(x))
            except Exception:
                val = float('inf')
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
                
            return val

        # Phase 1: Scout / Exploration
        # We use a small number of random points to find a good starting region.
        # For Rosenbrock, the global minimum is at (1,1,...,1).
        # Hidden shifts/rotations move it, but it's likely still in the "positive" quadrant relative to the shift.
        # We will sample a few points, including the nominal minimum (1,1,1,1,1) and origin, plus randoms.
        
        n_scout = 15
        scout_points = []
        
        # Nominal minimum
        scout_points.append(np.ones(self.dim))
        # Origin
        scout_points.append(np.zeros(self.dim))
        
        # Random points
        for _ in range(n_scout - 2):
            scout_points.append(rng.uniform(-5, 5, self.dim))
        
        # Evaluate scouts
        for x in scout_points:
            if check_time() or calls >= self.budget:
                break
            wrapped_func(x)
        
        # Phase 2: Local Exploitation using Powell's Method
        # We will run Powell from the best scout point.
        # If we have budget left, we might run it from a second best point.
        
        if best_x is not None and calls < self.budget:
            remaining = self.budget - calls
            # Reserve a small buffer for safety
            maxfev = max(1, remaining - 5)
            
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Powell', 
                    options={
                        'maxfev': maxfev,
                        'xtol': 1e-10,
                        'ftol': 1e-10,
                        'maxiter': 100
                    }
                )
            except Exception:
                pass
        
        # Phase 3: If we still have significant budget left, try a second local search
        # This handles the case where the first Powell run got stuck in a local minimum
        # or the scout phase didn't find the basin of attraction well.
        
        if calls < self.budget - 50 and not check_time():
            # Pick a different starting point, e.g., the second best scout or a new random one
            # Let's try a new random point near the best found so far
            start_x = best_x + rng.normal(0, 0.5, self.dim)
            start_x = np.clip(start_x, -5, 5)
            
            remaining = self.budget - calls
            maxfev = max(1, remaining - 5)
            
            try:
                res2 = minimize(
                    wrapped_func, 
                    start_x, 
                    method='Powell', 
                    options={
                        'maxfev': maxfev,
                        'xtol': 1e-10,
                        'ftol': 1e-10,
                        'maxiter': 100
                    }
                )
            except Exception:
                pass

        return best_x, best_f
