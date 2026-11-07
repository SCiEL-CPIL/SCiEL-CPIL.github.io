"""Hybrid optimizer for BBOB-style Rosenbrock f8: Random exploration + L-BFGS-B multi-start + (1+1)-ES polish."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start L-BFGS-B with global exploration via random restarts and ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Track best solution and value
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return best_f
            
            x_clip = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            calls_made += 1
            
            try:
                val = func(x_clip)
                val = float(val)
                if np.isfinite(val):
                    if val < best_f:
                        best_f = val
                        best_x = x_clip.copy()
                    return val
                else:
                    return 1e10
            except Exception:
                return 1e10

        # Phase 1: Random exploration to find promising regions
        # Use a fraction of budget for initial sampling
        n_random = 30
        rng = np.random.default_rng(42)

        candidates = []
        for i in range(n_random):
            if time.time() > deadline or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            candidates.append((fx, x.copy()))

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Select diverse seeds for local optimization
        max_seeds = 8
        selected_seeds = []
        for s in candidates:
            if len(selected_seeds) >= max_seeds:
                break
            diverse = True
            for ts in selected_seeds:
                if np.linalg.norm(s[1] - ts[1]) < 1.5:
                    diverse = False
                    break
            if diverse:
                selected_seeds.append(s)
        
        # Ensure the absolute best is always included
        if candidates and selected_seeds:
            if selected_seeds[0][0] != candidates[0][0]:
                selected_seeds[0] = candidates[0]

        # Phase 2: Local optimization from best candidates using L-BFGS-B
        # L-BFGS-B is generally more robust for Rosenbrock than SLSQP due to 
        # better handling of the narrow valley and bound constraints.
        
        for i, (sf, sx) in enumerate(selected_seeds):
            if time.time() > deadline or calls_made >= budget - 10:
                break
            
            remaining = budget - calls_made
            # Allocate more iterations to the first (best) seed
            if i == 0:
                maxiter = 100
            else:
                maxiter = max(20, remaining // (len(selected_seeds) - i + 1))
            
            try:
                result = minimize(
                    safe_func, 
                    sx, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': maxiter,
                        'maxfun': max(10, remaining - 5),
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                    }
                )
            except Exception:
                pass

        # Phase 3: Polish with (1+1)-ES
        # This helps fine-tune the solution if the gradient-based method stopped
        # slightly early or got stuck in a local minimum near the boundary.
        
        if best_x is not None and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            
            while calls_made < budget and time.time() < deadline - 5:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.05, 3.0)
                else:
                    sigma = max(sigma * 0.9, 1e-6)
                    
                if sigma < 1e-5:
                    break

        # Final check
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f
