"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-Start Powell Optimizer with final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 325.0  # 300s limit, 25s margin
        
        # Wrapper to count calls and enforce limits
        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return func(x)

        best_x = np.zeros(self.dim)
        best_fx = float('inf')
        
        # Better initial points: center + structured random + axis-aligned
        rng = np.random.default_rng(123)
        starts = [np.zeros(self.dim)]
        # Add points along axes to help with bent cigar (elongated valley)
        for axis in range(self.dim):
            for sign in [1, -1]:
                pt = np.zeros(self.dim)
                pt[axis] = sign * 3.0
                starts.append(pt)
        # Add some random points
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, self.dim))

        # Phase 1: Multi-start Powell with moderate tolerances
        for x0 in starts:
            if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                break
                
            try:
                res = minimize(
                    eval_func,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': 300,
                        'ftol': 1e-10,
                        'xatol': 1e-6
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except StopIteration:
                break
            except Exception:
                continue

        # Phase 2: Final high-precision polish from the best point found
        if self.calls < self.budget and time.time() - self.start_time < self.time_limit - 10:
            try:
                res = minimize(
                    eval_func,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 500,
                        'ftol': 1e-14,
                        'xatol': 1e-10
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        # Ensure we return something valid
        if best_x is None or not np.all(np.isfinite(best_x)):
             best_x = np.zeros(self.dim)
             if self.calls < self.budget:
                 best_fx = eval_func(best_x)
             else:
                 best_fx = 0.0
                 
        return best_x, best_fx
# EVOLVE-BLOCK-END
