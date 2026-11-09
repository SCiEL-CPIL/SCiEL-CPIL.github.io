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
        deadline = time.time() + 290  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return 1e10
            # Clip to box
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
        # Center
        starts.append(np.zeros(dim))
        # Random points in [-4, 4]
        for _ in range(20):
            starts.append(rng.uniform(-4, 4, dim))
        # Some points near boundaries
        for i in range(dim):
            pt = np.zeros(dim)
            pt[i] = 4.0
            starts.append(pt)
            pt[i] = -4.0
            starts.append(pt)

        # Shuffle starts
        perm = rng.permutation(len(starts))
        starts = [starts[i] for i in perm]

        # Allocate budget among restarts
        # We have 500 evals. Each minimize call uses some evals.
        # L-BFGS-B typically uses 10-50 evals to converge in 5D.
        # Let's try up to 10 restarts, with a max of 50 evals each initially, 
        # but let scipy manage the internal loop, we just count total.
        
        # Strategy: Run minimize from each start until time or budget runs out.
        # We can set 'maxiter' or 'maxfev' to limit per-restart, but we need to 
        # ensure we use the global budget.
        
        # Since we don't know how many evals each minimize will take, we will 
        # just run them sequentially and stop when call_count >= budget or time > deadline.
        
        # To avoid one restart eating all budget, we can limit maxfev per restart.
        # A reasonable maxfev per restart is 50. 500/50 = 10 restarts.
        
        maxfev_per_restart = 50
        
        for x0 in starts:
            if time.time() > deadline or call_count >= budget:
                break
            
            try:
                res = minimize(
                    safe_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfev': maxfev_per_restart}
                )
                if res.success or res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # If we have time and budget left, we can do a final local refinement from the best point
        if best_x is not None and time.time() < deadline and call_count < budget:
            try:
                res = minimize(
                    safe_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfev': budget - call_count}
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
