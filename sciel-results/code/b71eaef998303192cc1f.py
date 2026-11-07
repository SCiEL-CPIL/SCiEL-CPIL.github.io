"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with Nelder-Mead refinement for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Count evaluations
        calls = [0]
        
        def wrapped_func(x):
            calls[0] += 1
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))
        
        # Generate a set of starting points in [-5, 5]^dim
        # Use a mix of deterministic grid and random points
        # Strategy: center, axis-aligned points, and random points
        # Allocate budget: ~60% for L-BFGS-B multi-start, ~30% for NM refinement, ~10% reserve
        
        n_starts = 8  # Fixed number of starting points
        
        # Deterministic grid points
        grid_points = []
        # Center point
        grid_points.append(np.zeros(dim))
        # Points at ±2 in each dimension (10 points for dim=5)
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 2.0
                grid_points.append(p)
        
        # Add random points to fill up to n_starts
        while len(grid_points) < n_starts:
            grid_points.append(rng.uniform(-4, 4, dim))
        
        # If we have more than n_starts, randomly select
        if len(grid_points) > n_starts:
            indices = rng.choice(len(grid_points), n_starts, replace=False)
            grid_points = [grid_points[i] for i in indices]
        
        best_x = None
        best_f = np.inf
        
        # Allocate budget: 60% for L-BFGS-B, 30% for NM, 10% reserve
        lbb_budget = int(budget * 0.6)
        nm_budget = int(budget * 0.3)
        
        # Try L-BFGS-B from each starting point
        evals_per_run = max(15, lbb_budget // n_starts)
        
        for x0 in grid_points:
            if calls[0] >= lbb_budget:
                break
            
            x0 = np.clip(x0, -5.0, 5.0)
            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Run L-BFGS-B from this point
            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 200,
                        'maxfun': evals_per_run - 1,
                        'gtol': 1e-10,
                        'ftol': 1e-14,
                        'maxls': 50
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0)
            except Exception:
                pass
        
        # Nelder-Mead refinement from best point
        remaining = budget - calls[0]
        if remaining > 10 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            
            # Try Nelder-Mead with tight tolerances
            try:
                nm_result = minimize(
                    wrapped_func,
                    x,
                    method='Nelder-Mead',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 200,
                        'maxfev': remaining - 5,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                if nm_result.fun < best_f:
                    best_f = nm_result.fun
                    best_x = np.clip(nm_result.x, -5.0, 5.0)
            except Exception:
                pass
        
        # If we still have budget left, do a (1+1)-ES refinement
        remaining = budget - calls[0]
        if remaining > 5 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            
            # Adaptive step size (1+1)-ES refinement
            for _ in range(remaining - 1):
                if calls[0] >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = wrapped_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9
                
                # Stop if we've converged
                if sigma < 1e-8:
                    break
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Ensure we made at least one call
        if calls[0] == 0:
            x0 = np.zeros(dim)
            best_f = wrapped_func(x0)
            best_x = x0.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
