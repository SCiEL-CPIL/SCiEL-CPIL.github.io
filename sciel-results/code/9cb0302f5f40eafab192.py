"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Optimizer using Powell's Method (derivative-free coordinate-wise line search)
    with a Nelder-Mead fallback for final polishing.
    Designed for BBOB f14 (Different Powers) in 5 dimensions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0

        # Wrapper to count calls and clip to box
        def _eval(x):
            nonlocal calls
            if calls >= budget:
                raise RuntimeError("Budget exceeded")
            x_clipped = np.clip(x, -5.0, 5.0)
            calls += 1
            return func(x_clipped)

        # 1. Initial Point
        # f14 minimum is typically near the origin after standard transformations
        x = np.zeros(dim)
        try:
            fx = _eval(x)
        except Exception:
            return x, np.inf

        best_x = x.copy()
        best_f = fx

        # 2. Powell's Method
        # Powell's method works by searching along a set of directions.
        # It starts with the coordinate axes and updates them based on the best directions.
        # This is particularly effective for functions with separable or quasi-separable structure.
        
        # Initialize directions
        directions = np.eye(dim)
        
        # Tolerances for convergence
        tol_x = 1e-10
        tol_f = 1e-12
        
        # Main loop of Powell's method
        # We limit iterations to ensure we don't run out of budget or time
        max_powell_iters = 50
        
        for _ in range(max_powell_iters):
            if time.time() > deadline or calls >= budget:
                break
            
            # Check for convergence
            if np.linalg.norm(directions[0] - directions[1]) < tol_x:
                break
                
            x_min = x
            f_min = fx
            f_min_old = fx
            
            # Search along each direction
            for i in range(dim):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Find minimum along direction i
                # We use a simple line search: try positive and negative steps
                # Powell's original method uses a specific line search, but a simple
                # adaptive step size works well in low dims.
                
                # Determine initial step size
                # If this is the first direction, use a standard step
                # Otherwise, estimate from previous moves
                if i == 0:
                    step = 1.0
                else:
                    # Estimate step size based on previous direction's move
                    # directions[i] is the current direction
                    # We can look at the change in x
                    # A simpler heuristic: keep step size adaptive
                    step = 1.0 # Reset step size for simplicity, or use memory
                    
                # Line search along direction[i]
                # We perform a local minimization along the line x + t * direction[i]
                # Using a few evaluations to find the minimum
                
                # Simple 1D minimization using golden section or similar
                # To save calls, we use a small number of evaluations
                # Let's use a simple adaptive step approach:
                # 1. Try x + step * dir
                # 2. If better, try x + 2*step * dir, etc.
                # 3. If worse, try x - step * dir
                
                # Actually, scipy.optimize.minimize with method='Powell' is robust and handles
                # the line search internally. However, it might use too many calls if not careful.
                # Let's try using scipy's Powell but with strict call limits.
                
                # Since we are in a custom loop, let's implement a simple coordinate search
                # or rely on scipy for the whole thing if we can control it.
                
                # Given the complexity of a robust custom Powell, let's switch strategy:
                # Use scipy.optimize.minimize with method='Powell' as the primary engine,
                # with a tight maxiter and callback to stop if budget is near.
                # But scipy's Powell might not be as call-efficient as a custom one in 5D.
                
                # Let's stick to the custom loop but make the line search efficient.
                # For each direction, we do a small number of evaluations.
                
                # Simple 1D search:
                # We know f at x.
                # Let d = directions[i]
                # Try t = step
                # If f(x + t*d) < f(x), move to x + t*d, increase step
                # If f(x + t*d) > f(x), try t = -step
                # ...
                
                # This is getting complex. Let's use a simpler, robust approach:
                # Coordinate-wise minimization (Cyclic Coordinate Descent) with adaptive steps.
                # This is a variant of Powell's that is simpler to implement safely.
                
                # For each coordinate i:
                #   Perform a 1D minimization along axis i.
                
                d = np.zeros(dim)
                d[i] = 1.0
                
                # 1D minimization along d
                # Use a small number of evaluations (e.g., 3-5) to find a better point
                # We use a simple golden-section-like search or just a few probes.
                
                # Probe 1: x + 0.5 * d
                # Probe 2: x - 0.5 * d
                # Choose the best, then refine.
                
                # To be very efficient, we can just use scipy's minimize on a 1D problem
                # but that has overhead.
                
                # Let's use a simple step-halving approach.
                # Current best at this direction is x.
                # Try x + 1.0 * d
                # If better, try x + 2.0 * d, etc.
                # If worse, try x - 1.0 * d
                
                # This is essentially a (1+1)-ES on each coordinate sequentially.
                # This is very safe and respects the budget.
                
                # Let's implement this "Cyclic Adaptive Coordinate Descent"
                
                # Step size for this coordinate
                # We can maintain a step size per coordinate
                if not hasattr(self, '_step_sizes'):
                    self._step_sizes = np.ones(dim)
                    
                step = self._step_sizes[i]
                
                # Try positive step
                x_plus = x + step * d
                f_plus = _eval(x_plus)
                
                if f_plus < fx:
                    x = x_plus
                    fx = f_plus
                    self._step_sizes[i] *= 1.5 # Increase step if moving
                else:
                    # Try negative step
                    x_minus = x - step * d
                    f_minus = _eval(x_minus)
                    if f_minus < fx:
                        x = x_minus
                        fx = f_minus
                        self._step_sizes[i] *= 1.5
                    else:
                        # No improvement, decrease step
                        self._step_sizes[i] *= 0.5
                        # If step becomes too small, we can skip further refinement for this coord
                        if self._step_sizes[i] < 1e-6:
                            continue
                
                # Update best if improved
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # End of coordinate loop
            # Check convergence of the full cycle
            if abs(f_min_old - fx) < tol_f:
                break

        # 3. Final Polish with Nelder-Mead
        # If we have budget left, use NM to tighten the solution.
        remaining = budget - calls
        if remaining > 10:
            try:
                from scipy.optimize import minimize
                
                nm_calls = [0]
                def nm_func(z):
                    nonlocal calls
                    if calls >= budget:
                        return 1e10 # Penalize budget overrun
                    z_clipped = np.clip(z, -5.0, 5.0)
                    calls += 1
                    nm_calls[0] += 1
                    return func(z_clipped)
                
                maxiter = max(5, min(100, remaining // 2))
                res = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                
                if res.fun < best_f:
                    best_x = np.clip(res.x, -5.0, 5.0)
                    best_f = res.fun
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
