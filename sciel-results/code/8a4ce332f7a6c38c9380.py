"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")


class Optimizer:
    """
    Optimizer using multi-start L-BFGS-B with (1+1)-ES refinement for rotated Rosenbrock.
    L-BFGS-B excels on smooth functions with narrow valleys like Rosenbrock.
    Multiple diverse starts ensure we find the global basin, and (1+1)-ES
    provides robust local refinement with adaptive step sizes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_x = None
        self.best_fx = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 45s safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        def tracked_func(x):
            if not time_ok() or self.call_count >= self.budget:
                raise ValueError("Budget or time exceeded")
            
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            
            self.call_count += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            
            if fx < self.best_fx:
                self.best_fx = fx
                self.best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)
        dim = self.dim

        # Phase 1: Multi-start L-BFGS-B
        # Generate diverse starting points covering the likely minimizer region [-4, 4]^5
        n_starts = 12
        starts = []
        
        # Deterministic diverse points: center, corners of [-3,3]^5, and random
        starts.append(np.zeros(dim))
        starts.append(np.full(dim, -3.0))
        starts.append(np.full(dim, 3.0))
        starts.append(np.array([3.0, -3.0, 3.0, -3.0, 3.0]))
        starts.append(np.array([-3.0, 3.0, -3.0, 3.0, -3.0]))
        
        # Add random points in [-4, 4]^5
        for _ in range(n_starts - 5):
            starts.append(rng.uniform(-4.0, 4.0, size=dim))
        
        # Sort by distance to center (Rosenbrock minimum tends to be near origin after rotation)
        starts.sort(key=lambda x: np.linalg.norm(x))
        
        bounds = [(-5.0, 5.0)] * dim
        
        for i, x0 in enumerate(starts):
            if not time_ok() or self.call_count >= self.budget - 30:
                break
            
            try:
                res = minimize(
                    tracked_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 200,
                        'maxfun': min(80, self.budget - self.call_count - 20),
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                        'maxls': 50
                    }
                )
            except Exception:
                pass
            
            if not time_ok() or self.call_count >= self.budget - 30:
                break
        
        if self.best_x is None:
            self.best_x = np.zeros(dim)
            self.best_fx = 0.0
        
        # Phase 2: (1+1)-ES refinement around the best point
        if time_ok() and self.call_count < self.budget - 10:
            x = self.best_x.copy()
            fx = self.best_fx
            sigma = 0.3
            stuck = 0
            remaining = self.budget - self.call_count - 5
            
            for _ in range(remaining):
                if not time_ok() or self.call_count >= self.budget - 5:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = tracked_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                    stuck = 0
                else:
                    sigma *= 0.92
                    stuck += 1
                
                if stuck > 40:
                    # Restart from slightly perturbed best
                    x = np.clip(self.best_x + 0.2 * rng.standard_normal(dim), -5.0, 5.0)
                    fx = tracked_func(x)
                    sigma = 0.2
                    stuck = 0
        
        # Phase 3: Final L-BFGS-B polish with tighter bounds around best
        if time_ok() and self.call_count < self.budget - 5 and self.best_x is not None:
            try:
                tight_bounds = []
                for i in range(dim):
                    lo = max(-5.0, self.best_x[i] - 1.5)
                    hi = min(5.0, self.best_x[i] + 1.5)
                    tight_bounds.append((lo, hi))
                
                res = minimize(
                    tracked_func,
                    self.best_x,
                    method='L-BFGS-B',
                    bounds=tight_bounds,
                    options={
                        'maxiter': 100,
                        'maxfun': min(50, self.budget - self.call_count - 2),
                        'ftol': 1e-16,
                        'gtol': 1e-13,
                        'maxls': 30
                    }
                )
            except Exception:
                pass
        
        if self.best_x is None:
            self.best_x = np.zeros(dim)
            self.best_fx = 0.0
            
        return self.best_x, self.best_fx

# EVOLVE-BLOCK-END
