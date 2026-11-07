# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Portfolio optimizer for BBOB f7 (Step ellipsoidal) on [-5,5]^d."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Count function calls
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

        def count_eval(x):
            self._call_count += 1
            val = func(x)
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # Strategy 1: L-BFGS-B from multiple diverse starting points
        # The step ellipsoidal function has a global minimum at some shifted point
        # Try several starting points and use L-BFGS-B to converge
        
        # Generate diverse starting points
        n_starts = 10
        starts = []
        
        # Origin
        starts.append(np.zeros(self.dim))
        
        # Random points in the box
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, self.dim))
        
        # Also try some points near the boundaries of the likely minimum region
        for _ in range(5):
            starts.append(rng.uniform(-3, 3, self.dim))
        
        # Shuffle starts
        perm = rng.permutation(len(starts))
        starts = [starts[i] for i in perm]
        
        for x0 in starts:
            if self._call_count >= self.budget - 50:
                break
            if time.time() > deadline:
                break
            
            try:
                res = minimize(count_eval, x0, method='L-BFGS-B',
                             bounds=[(-5, 5)] * self.dim,
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue
        
        # Strategy 2: CMA-ES-like exploration with adaptive step size
        if self._call_count < self.budget - 20 and time.time() < deadline:
            # Use a simple evolutionary strategy for the remaining budget
            if self._best_x is None:
                x = rng.uniform(-5, 5, self.dim)
            else:
                x = self._best_x.copy()
            fx = self._best_f
            sigma = 1.0
            
            while self._call_count < self.budget - 10:
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = count_eval(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.9
                
                # If sigma becomes too small, try a larger jump
                if sigma < 0.01:
                    sigma = 2.0
        
        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
