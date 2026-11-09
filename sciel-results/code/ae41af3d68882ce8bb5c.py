"""Optimizer for BBOB f8 (Rosenbrock) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        call_count = [0]
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal best_x, best_f
            if call_count[0] >= budget:
                return best_f if best_f is not None else 1e10
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            call_count[0] += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Diverse Initialization ---
        # We want to cover the space well.
        # 1. Origin
        # 2. Grid points at +/- 1, +/- 2, +/- 3
        # 3. Random points
        
        init_points = []
        init_points.append(np.zeros(dim))
        
        # Grid
        for v in [-3, -2, -1, 1, 2, 3]:
            init_points.append(np.full(dim, v))
            
        # Random
        rng = np.random.default_rng(123)
        for _ in range(15):
            init_points.append(rng.uniform(-4.5, 4.5, size=dim))
            
        # Axis-aligned
        for i in range(dim):
            for v in [-2, 2]:
                p = np.zeros(dim)
                p[i] = v
                init_points.append(p)
                
        # Evaluate all init points
        for p in init_points:
            if call_count[0] >= budget - 50: # Leave some for optimization
                break
            if time.time() > deadline:
                break
            safe_func(p)
            
        if best_x is None:
            best_x = np.zeros(dim)
            safe_func(best_x)

        # --- Phase 2: Multi-Start Local Optimization ---
        # We will try to optimize from the best few initial points.
        # Powell is robust for Rosenbrock.
        
        # Get sorted list of initial points by their f-value
        # We didn't store them all, so we just re-evaluate or use the best found.
        # Actually, we can just restart Powell from the current best, 
        # and if it fails or improves, we continue.
        # But to be safe against local minima (though Rosenbrock has 1), 
        # let's try a few restarts from the best initial points.
        
        # Since we didn't store the f-values of all init points in a list, 
        # let's just run Powell from the current best_x.
        # If it doesn't improve much, we might be stuck.
        
        # Strategy: Run Powell from best_x.
        try:
            res = minimize(
                safe_func,
                best_x,
                method='Powell',
                bounds=[(-5.0, 5.0)] * dim,
                options={'maxiter': 100, 'maxfev': 150} # Limit calls
            )
            # safe_func already updates best_x/best_f if it finds something better
            # But minimize returns the result. We should check if it's better.
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass

        # If we still have budget, try L-BFGS-B for precision
        if call_count[0] < budget - 20 and time.time() < deadline:
            try:
                res2 = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 200, 'maxfun': 200}
                )
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x
            except Exception:
                pass

        # Final Polish with Nelder-Mead if budget allows
        if call_count[0] < budget - 10 and time.time() < deadline:
            try:
                res3 = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxfev': 100, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res3.fun < best_f:
                    best_f = res3.fun
                    best_x = res3.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
