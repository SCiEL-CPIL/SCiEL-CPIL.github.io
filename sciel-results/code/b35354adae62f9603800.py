"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Global-Local Optimizer using Coordinate Descent with Brent for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        n_calls = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def check_time():
            return time.time() < deadline and n_calls < budget

        def evaluate(x):
            nonlocal n_calls, best_x, best_f
            if not check_time():
                return None
            x = np.asarray(x, dtype=float)
            if x.shape != (d,) or not np.all(np.isfinite(x)):
                return None
            # Clip to box to ensure validity, although func clips too
            x = np.clip(x, -5, 5)
            fx = func(x)
            n_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Exploration
        # Sample diverse points to find a good starting point for local search
        n_global = 20
        for _ in range(n_global):
            if not check_time():
                break
            x = rng.uniform(-5, 5, d)
            evaluate(x)

        # Phase 2: Coordinate Descent with Brent's Method
        # For each coordinate, perform a 1D minimization using Brent's method
        # This is effective for separable-like structures and can achieve high precision
        
        def brent_minimize_1d(x0, axis, f_val_at_x0):
            """
            Perform 1D minimization along 'axis' starting from x0.
            Returns (new_x, new_f) or (None, None) if failed.
            """
            nonlocal n_calls, best_x, best_f
            if not check_time():
                return None, None
            
            # Define the 1D function
            def f_1d(t):
                x = x0.copy()
                x[axis] = t
                fx = evaluate(x)
                if fx is None:
                    return np.inf
                return fx
            
            # Determine search interval around x0[axis]
            # Start with a small step and expand if needed
            step = 0.5
            a = x0[axis] - step
            b = x0[axis] + step
            
            # Clip to bounds
            a = max(-5, min(5, a))
            b = max(-5, min(5, b))
            
            # Evaluate at endpoints and center
            fa = f_1d(a)
            fb = f_1d(b)
            fc = f_1d(x0[axis])
            
            # If we are out of budget, return current best
            if not check_time():
                return None, None
            
            # Simple bracketing: if fc is not the min, expand
            # For simplicity, we'll use a fixed interval and rely on Brent
            # Brent's method requires a bracket where the minimum is inside
            # Let's try to find a better bracket
            
            # Try expanding the interval if needed
            for _ in range(3):
                if not check_time():
                    return None, None
                if fc < fa and fc < fb:
                    break
                elif fa < fc and fa < fb:
                    b = x0[axis]
                    fb = fc
                    step *= 2
                    a = x0[axis] - step
                    a = max(-5, min(5, a))
                    fa = f_1d(a)
                else: # fb < fc and fb < fa
                    a = x0[axis]
                    fa = fc
                    step *= 2
                    b = x0[axis] + step
                    b = max(-5, min(5, b))
                    fb = f_1d(b)
            
            if not check_time():
                return None, None
            
            # Use scipy.optimize.minimize_scalar with method='brent'
            try:
                from scipy.optimize import minimize_scalar
                res = minimize_scalar(f_1d, bounds=(a, b), method='brent', options={'maxiter': 20})
                if res.success:
                    x_new = x0.copy()
                    x_new[axis] = res.x
                    f_new = res.fun
                    if f_new < f_val_at_x0:
                        return x_new, f_new
                    else:
                        return x0, f_val_at_x0
                else:
                    return x0, f_val_at_x0
            except Exception:
                return x0, f_val_at_x0

        # Main coordinate descent loop
        # We'll perform multiple passes over the coordinates
        # Each pass does one Brent minimization per coordinate
        max_passes = 100
        for pass_num in range(max_passes):
            if not check_time():
                break
            
            improved = False
            for axis in range(d):
                if not check_time():
                    break
                
                x0 = best_x.copy()
                f0 = best_f
                
                x_new, f_new = brent_minimize_1d(x0, axis, f0)
                
                if x_new is not None and f_new is not None:
                    if f_new < best_f - 1e-10:
                        best_x = x_new
                        best_f = f_new
                        improved = True
                    else:
                        # If no improvement, update best_x to x0 (which is same as best_x)
                        pass
            
            # If no improvement in a full pass, break
            if not improved:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
