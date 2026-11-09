"""Multi-restart SciPy optimizer for BBOB f9 (Rosenbrock rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart optimizer for minimizing func on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return 1e10
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            call_count += 1
            return val

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        starts = []
        
        # 1. Center
        starts.append(np.zeros(dim))
        
        # 2. Points along main diagonal (valley direction for unrotated Rosenbrock)
        for scale in [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5]:
            starts.append(np.full(dim, scale))
            
        # 3. Axis-aligned points (exploring each dimension)
        for i in range(dim):
            for val in [1.0, -1.0, 2.0, -2.0, 0.0]:
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)
                
        # 4. Random points in a slightly larger box to ensure coverage
        for _ in range(20):
            starts.append(rng.uniform(-4.5, 4.5, dim))
            
        # 5. Points near the corners of the search space
        for corner in [(-4, -4, -4, -4, -4), (4, 4, 4, 4, 4), (-4, 4, -4, 4, -4), (4, -4, 4, -4, 4)]:
            starts.append(np.array(corner))

        # Shuffle starts to randomize order
        perm = rng.permutation(len(starts))
        starts = [starts[i] for i in perm]

        # Adaptive restart strategy
        # Start with a modest budget per restart
        current_maxfev = 20
        min_maxfev = 10
        max_maxfev = 60

        for x0 in starts:
            if time.time() > deadline or call_count >= budget:
                break
            
            # If we have a very good solution, skip distant starts to save budget
            if best_f < 0.1 and best_x is not None:
                dist = np.linalg.norm(x0 - best_x)
                if dist > 3.0:
                    continue

            try:
                res = minimize(
                    safe_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfev': current_maxfev, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    # Made progress, allow more budget for next restart
                    current_maxfev = min(max_maxfev, int(current_maxfev * 1.2))
                    update_best(res.x, res.fun)
                else:
                    # No improvement, reduce budget for next restart
                    current_maxfev = max(min_maxfev, int(current_maxfev * 0.8))
            except Exception:
                pass

        # Final local refinement from the best point with remaining budget
        if best_x is not None and time.time() < deadline and call_count < budget:
            remaining = budget - call_count
            if remaining > 5:
                try:
                    # Use L-BFGS-B for high precision refinement
                    res = minimize(
                        safe_func, 
                        best_x, 
                        method='L-BFGS-B', 
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 500, 'maxfev': remaining, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                except Exception:
                    pass

        # If budget remains, try Nelder-Mead for a different perspective
        if best_x is not None and time.time() < deadline and call_count < budget:
            remaining = budget - call_count
            if remaining > 20:
                try:
                    res = minimize(
                        safe_func, 
                        best_x, 
                        method='Nelder-Mead', 
                        options={'maxiter': 500, 'maxfev': remaining, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                except Exception:
                    pass

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
