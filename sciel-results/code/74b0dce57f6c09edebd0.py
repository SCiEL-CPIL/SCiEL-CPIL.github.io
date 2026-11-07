"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Hybrid optimizer:
    1. Powell's method for anisotropic alignment (handles the 'cigar' shape).
    2. L-BFGS-B for fast, precise convergence near the minimum.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        dim = self.dim
        budget = self.budget

        # State tracking
        best_x = np.zeros(dim)
        best_fx = np.inf
        calls = 0

        def wrapped_func(x):
            nonlocal calls, best_x, best_fx
            if calls >= budget:
                return best_fx
            if time.time() > deadline:
                return best_fx
            
            x = np.asarray(x, dtype=float)
            if not np.all(np.isfinite(x)):
                return 1e10
            
            # Clip to bounds to be safe, though scipy handles bounds
            x_clipped = np.clip(x, -5.0, 5.0)
            
            fx = float(func(x_clipped))
            calls += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
            
            return fx

        # Initial point: Center of the box
        x0 = np.zeros(dim)
        
        # Evaluate initial point
        wrapped_func(x0)

        # Phase 1: Powell's Method
        # Powell's method is derivative-free and excellent for anisotropic functions
        # like Bent Cigar because it rotates the search directions to align with the valley.
        # We allocate ~40% of the budget to this phase.
        powell_budget = int(budget * 0.4)
        
        if calls < budget and time.time() < deadline:
            try:
                bounds = [(-5.0, 5.0)] * dim
                res_powell = minimize(
                    wrapped_func,
                    best_x,
                    method='Powell',
                    bounds=bounds,
                    options={
                        'maxiter': 100, 
                        'maxfev': powell_budget, 
                        'xtol': 1e-6, 
                        'ftol': 1e-6
                    }
                )
            except Exception:
                pass

        # Phase 2: L-BFGS-B
        # Use the best point from Powell as the start for L-BFGS-B.
        # L-BFGS-B will converge very quickly from a well-aligned starting point.
        remaining_budget = budget - calls
        if remaining_budget > 10 and time.time() < deadline:
            try:
                bounds = [(-5.0, 5.0)] * dim
                res_lbfgs = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100, 
                        'maxfun': remaining_budget, 
                        'gtol': 1e-8,
                        'ftol': 1e-12
                    }
                )
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
