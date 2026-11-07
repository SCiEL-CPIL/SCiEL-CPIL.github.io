"""Exploit quadratic structure of Discus via BFGS optimization with robust initial exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Optimizer that exploits the quadratic structure of BBOB f11 (Discus)
    using BFGS, which is highly effective for quadratic functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 330s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls
            if calls >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Exploration
        # We need to find the "needle" direction.
        # Evaluate origin and points along axes.
        evaluate(np.zeros(dim))
        
        # Axis sweeps to identify the flat direction
        for axis in range(dim):
            for r in [1.0, 2.0, 3.0, 4.0]:
                if calls >= budget or time.time() > deadline:
                    break
                x_pos = np.zeros(dim)
                x_pos[axis] = r
                evaluate(x_pos)
                
                x_neg = np.zeros(dim)
                x_neg[axis] = -r
                if calls < budget and time.time() < deadline:
                    evaluate(x_neg)
            if calls >= budget or time.time() > deadline:
                break

        # Random sweeps to catch rotated features
        n_random = 20
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        # Phase 2: BFGS Optimization
        # BFGS is ideal for quadratic functions. It converges in O(n) steps.
        if best_x is not None and calls < budget:
            # Run BFGS from the best point found
            # Use a callback to track calls and stop if we exceed budget or deadline
            def callback(xk):
                return calls >= budget or time.time() > deadline

            try:
                res = minimize(
                    evaluate, 
                    best_x, 
                    method='BFGS', 
                    options={'maxiter': 50, 'maxfev': max(1, budget - calls), 'disp': False}
                )
                # Update best if BFGS found something better (it should)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Phase 3: Polish with a second BFGS run if budget remains
        # Sometimes BFGS can get stuck in a non-smooth region of the max() term.
        # A second run from a slightly perturbed point can help.
        if calls < budget - 10 and time.time() < deadline:
            x_start = best_x + rng.normal(0, 0.1, dim)
            x_start = np.clip(x_start, -5.0, 5.0)
            
            try:
                res2 = minimize(
                    evaluate, 
                    x_start, 
                    method='BFGS', 
                    options={'maxiter': 50, 'maxfev': max(1, budget - calls), 'disp': False}
                )
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
