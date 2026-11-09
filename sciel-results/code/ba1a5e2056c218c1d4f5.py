"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-Start Powell Optimizer for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        # 300s limit, use 280s to be safe.
        self.time_limit = 280.0 
        
        d = self.dim
        N = self.budget
        
        # Wrapper to count calls and enforce limits
        def eval_func(x):
            if self.calls >= N:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return func(x)

        best_x = np.zeros(d)
        best_fx = float('inf')
        
        # Initialize best_x with a valid point
        # We will evaluate the first start point immediately to ensure best_x is valid
        rng = np.random.default_rng(42)
        
        # Generate a set of starting points.
        # 0 is a good default for BBOB if shift is small, but shift is hidden.
        # Random points cover the space.
        # Powell is robust, so even random starts usually work well on f12.
        starts = []
        starts.append(np.zeros(d))
        for _ in range(5):
            starts.append(rng.uniform(-4.5, 4.5, d))

        for x0 in starts:
            # Check limits before starting a new phase
            if self.calls >= N:
                break
            if time.time() - self.start_time > self.time_limit:
                break
                
            try:
                # Powell method is excellent for anisotropic functions like Bent Cigar
                # It adapts its search directions to the curvature.
                res = minimize(
                    eval_func,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': 200,  # Limit iterations to prevent long hangs
                        'ftol': 1e-12,   # Tight tolerance for precision
                        'xatol': 1e-8,   # Tight position tolerance
                        'disp': False
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
                    
            except StopIteration:
                # Budget or time exceeded, break out of loop
                break
            except Exception:
                # If Powell crashes for any reason (e.g., NaN, convergence issues), 
                # just continue to the next start point.
                continue

        # If no valid result was found (e.g. all exceptions), try a single random eval
        if not np.isfinite(best_fx):
            if self.calls < N and time.time() - self.start_time < self.time_limit:
                x0 = np.zeros(d)
                try:
                    best_fx = eval_func(x0)
                    best_x = x0
                except:
                    best_fx = 0.0
                    best_x = np.zeros(d)
            else:
                best_fx = 0.0
                best_x = np.zeros(d)

        return best_x, best_fx
# EVOLVE-BLOCK-END
