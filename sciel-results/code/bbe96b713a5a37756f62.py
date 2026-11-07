"""Rewritten optimizer for BBOB-style Rosenbrock (f8) with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with deadline guard for Rosenbrock-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit

        # Track best solution found
        best_x = np.zeros(self.dim)
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            call_count += 1
            if call_count > self.budget:
                return np.inf
            val = func(np.asarray(x))
            return float(val)

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        
        # Strategy: combine random starts, grid-like points, and points near 0
        starts = []
        
        # Point near origin (common for Rosenbrock)
        starts.append(np.zeros(self.dim))
        
        # Points along the "valley" direction of Rosenbrock
        for i in range(5):
            pt = np.zeros(self.dim)
            pt[i] = 1.0
            starts.append(pt)
        
        # Random points in the box
        for i in range(20):
            starts.append(rng.uniform(-4, 4, self.dim))
        
        # Points with mixed positive/negative
        for i in range(10):
            starts.append(rng.choice([-1, 1], size=self.dim) * rng.uniform(0.5, 3.0, self.dim))
        
        # Shuffle starts
        np.random.shuffle(starts)

        for start_pt in starts:
            # Check time
            if time.time() > deadline:
                break
            if call_count >= self.budget:
                break

            # Clip start to box
            x0 = np.clip(start_pt, -5, 5)
            
            # Check if we've already evaluated this point
            fx0 = safe_func(x0)
            if not np.isfinite(fx0):
                continue
            
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            # Run Nelder-Mead from this start
            try:
                result = minimize(
                    safe_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'maxfev': 100,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass

            # Check if we're close to convergence
            if best_f < 1e-10:
                break

        # Final refinement: if we have budget left, do a local search around best_x
        if time.time() < deadline and call_count < self.budget - 50:
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50,
                        'maxfev': 50,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
