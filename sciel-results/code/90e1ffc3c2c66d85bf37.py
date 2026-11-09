# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def sobol_sequence(n, dim, seed=0):
    """
    Generate a deterministic set of space-filling points using a simple 
    Halton-like sequence or a fixed set of well-distributed points.
    Since SciPy doesn't have a direct 'sobol' in older versions without 
    specific imports or newer versions, we use a simple deterministic 
    grid/Halton approach to ensure distinct, spread-out starts.
    For dim=5 and n=5, this is sufficient.
    """
    # Simple Halton sequence generator
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = []
    # We want n points in dim dimensions
    # To ensure they are distinct and spread out, we offset the index
    for i in range(n):
        p = []
        for d in range(dim):
            base = d + 2  # 2, 3, 5, 7, 11 for dim 5
            # Offset index to avoid 0 and ensure spread
            val = halton(i + 1, base)
            # Map [0, 1) to [-5, 5)
            p.append(val * 10 - 5)
        points.append(p)
    return np.array(points)

class Optimizer:
    """Multi-start Nelder-Mead with Sobol-like initialization for f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        def time_ok():
            return time.time() < deadline - 2.0

        # 1. Generate initial candidates using space-filling sequence
        n_starts = 5
        init_points = sobol_sequence(n_starts, dim)
        
        best_x = init_points[0]
        best_f = np.inf
        calls_made = 0

        # Evaluate all initial points
        candidates = []
        for p in init_points:
            if calls_made >= budget:
                break
            x_eval = np.clip(p, -5, 5)
            f_val = func(x_eval)
            calls_made += 1
            candidates.append((f_val, x_eval))
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_eval.copy()

        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])

        # 2. Run Nelder-Mead from the best few starts
        # We will run NM on the top 3 candidates if budget allows
        nm_budget_per_run = max(10, (budget - calls_made) // 3)
        
        for i in range(min(3, len(candidates))):
            if not time_ok() or calls_made >= budget:
                break
            
            current_f, current_x = candidates[i]
            
            # Define the objective function for scipy
            def objective(x):
                nonlocal calls_made
                if calls_made >= budget:
                    return np.inf
                if not time_ok():
                    return np.inf
                # Clip to box
                x_clip = np.clip(x, -5, 5)
                val = func(x_clip)
                calls_made += 1
                return val

            # Nelder-Mead options
            # maxfev is handled by our budget check in objective, but we set a reasonable limit
            # adaptive=True allows step size adjustment
            options = {
                'maxfev': nm_budget_per_run,
                'xatol': 1e-6,
                'fatol': 1e-6,
                'adaptive': True
            }
            
            try:
                res = minimize(objective, current_x, method='Nelder-Mead', options=options)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # 3. If budget remains, do a final local polish with (1+1)-ES or NM from the best point
        if calls_made < budget - 5 and time_ok():
             # Use a small step size ES for final polish
             sigma = 0.1
             x = best_x.copy()
             fx = best_f
             
             for _ in range(budget - calls_made):
                 if not time_ok() or calls_made >= budget:
                     break
                 y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                 fy = func(y)
                 calls_made += 1
                 
                 if fy < fx:
                     x = y
                     fx = fy
                     if fx < best_f:
                         best_f = fx
                         best_x = x.copy()
                     sigma *= 1.2
                 else:
                     sigma *= 0.8
                     sigma = max(sigma, 1e-6)

        return best_x, best_f
# EVOLVE-BLOCK-END
