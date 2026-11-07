import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = time.time()
        # Deadline guard: 300s limit, keep 10s margin for safety
        self.deadline = self.start_time + 290.0

    def _check_time(self):
        return time.time() < self.deadline

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Wrap func to track calls and enforce budget
        # Note: The task says "Every call is recorded". 
        # If we wrap, we must ensure we don't exceed budget.
        # The evaluator calls __call__(func). 
        # We will manage the calls internally.
        
        def safe_func(x):
            if self.calls >= self.budget:
                return self.best_f
            if not self._check_time():
                return self.best_f
            x = np.asarray(x, dtype=float)
            # Clip to box to ensure validity, though func clips internally
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            self.calls += 1
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Strategy: Multi-Start Nelder-Mead
        # Lunacek has a global minimum and a local minimum.
        # Nelder-Mead is good for local refinement.
        # We need to start from many points to hit the global basin.
        
        # Budget allocation:
        # Each Nelder-Mead call uses roughly (dim + 1) + iterations * dim calls.
        # For dim=5, initial simplex is 6 points.
        # Let's estimate ~50-100 calls per successful local optimization.
        # With 500 budget, we can afford ~5-10 good restarts.
        # Let's try 10 restarts.
        
        n_restarts = 10
        
        # Generate initial points
        # 1. Origin
        # 2-10. Random uniform in [-5, 5]^dim
        # Also, maybe some points near the known structure of Lunacek?
        # Lunacek global min is often near 0, local min near 2.5 or so.
        # But we don't know the shift/rotation.
        # Random uniform is safest.
        
        initial_points = []
        initial_points.append(np.zeros(self.dim))
        
        # Add random points
        for _ in range(n_restarts - 1):
            initial_points.append(rng.uniform(-5, 5, self.dim))
            
        # Shuffle to ensure origin isn't always first (though order matters less with multi-start)
        # Actually, keeping origin first might be good if it's often good.
        # Let's just iterate through them.
        
        # We will run Nelder-Mead for each start point.
        # We need to stop if budget runs out or time is up.
        
        for i in range(n_restarts):
            if self.calls >= self.budget:
                break
            if not self._check_time():
                break
            
            x0 = initial_points[i]
            
            # Nelder-Mead options
            # maxiter: maximum number of iterations
            # We want to limit the number of function calls per restart.
            # Since we don't have a direct call limit in scipy's minimize,
            # we rely on the safe_func returning early if budget is hit.
            # However, if safe_func returns best_f (a constant) for all new points,
            # Nelder-Mead might stall or behave weirdly.
            # Better to just let it run until it converges or hits a maxiter that corresponds to ~budget/10 calls.
            
            # A typical Nelder-Mead run for dim=5 might take 20-50 iterations.
            # Each iteration is ~dim calls. So 50*5 = 250 calls.
            # That's too many for 10 restarts (2500 calls).
            # We need to limit maxiter per restart.
            # 500 calls / 10 restarts = 50 calls per restart.
            # 50 calls / 5 dim = 10 iterations.
            # So maxiter=10 per restart.
            
            options = {
                'maxiter': 10,
                'xatol': 1e-4,
                'fatol': 1e-4
            }
            
            try:
                res = minimize(
                    safe_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options=options
                )
            except Exception:
                # If minimize crashes, continue to next restart
                pass
            
            # Check if we made progress
            # safe_func updates self.best_f and self.best_x automatically
            
        # If we still have budget left, do a final polish on the best point found
        if self.calls < self.budget and self._check_time() and self.best_x is not None:
            # Run Nelder-Mead again on the best point with more iterations
            x0 = self.best_x
            options = {
                'maxiter': 50, # Use remaining budget
                'xatol': 1e-6,
                'fatol': 1e-6
            }
            try:
                res = minimize(
                    safe_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options=options
                )
            except Exception:
                pass

        return self.best_x, self.best_f
