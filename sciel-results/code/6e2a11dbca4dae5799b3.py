"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Deterministic multi-scale coordinate descent for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        call_count = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Deterministic multi-scale coordinate descent
        # We scan each dimension with geometrically shrinking step sizes.
        # This is different from stochastic population-based methods.
        
        # Initial point: center of the box
        x = np.zeros(dim)
        f_x = wrapped_func(x)
        
        # Step sizes: start large, shrink geometrically
        # We want to cover the range [-5, 5] initially.
        # Step sizes: 10, 5, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125, 0.0390625, ...
        # We will iterate through these scales.
        
        # To ensure early progress, we start with a coarse scan.
        # For each scale, we do a coordinate descent pass.
        # If we find a better point, we continue with that scale or move to next.
        # A simple deterministic approach:
        # For scale in scales:
        #   For each dimension d:
        #     Try x[d] +/- step
        #     If better, accept and maybe try again in same direction (line search)
        
        # Let's implement a simple version:
        # 1. Coarse grid scan? No, too many points.
        # 2. Coordinate descent with adaptive steps.
        
        # Scales:
        scales = [5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125, 0.0390625, 0.01953125, 0.009765625, 0.0048828125, 0.00244140625]
        
        for step in scales:
            if call_count >= budget - 50:  # Leave some for final polish
                break
            if time.time() > deadline - 10:
                break
                
            # Coordinate descent pass
            for d in range(dim):
                if call_count >= budget - 50:
                    break
                if time.time() > deadline - 10:
                    break
                
                # Try positive step
                x_plus = x.copy()
                x_plus[d] += step
                x_plus = np.clip(x_plus, -5, 5)
                f_plus = wrapped_func(x_plus)
                
                if f_plus < f_x:
                    x = x_plus
                    f_x = f_plus
                    # Maybe do a small line search in this direction?
                    # For now, just accept and move to next dimension.
                    # To be more thorough, we could try to go further.
                    # But simple CD is deterministic and predictable.
                
                # Try negative step
                x_minus = x.copy()
                x_minus[d] -= step
                x_minus = np.clip(x_minus, -5, 5)
                f_minus = wrapped_func(x_minus)
                
                if f_minus < f_x:
                    x = x_minus
                    f_x = f_minus

        # Final polish with a local optimizer if budget remains
        remaining_budget = budget - call_count
        if remaining_budget > 20:
            try:
                # Use Nelder-Mead for local refinement
                # maxiter is not strictly function evaluations, but we can monitor
                # We'll use a small maxiter to be safe
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
