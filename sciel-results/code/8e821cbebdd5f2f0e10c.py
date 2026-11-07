"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: Random exploration + L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x, best_f = x.copy(), f

        # Phase 1: Random exploration to find diverse seeds
        n_explore = 10
        seeds = []
        
        for i in range(n_explore):
            if not time_ok() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            try_update(x, fx)
            seeds.append((x.copy(), fx))

        # Sort seeds by function value
        seeds.sort(key=lambda s: s[1])
        
        # Keep top diverse seeds (at most 6)
        max_seeds = 6
        if len(seeds) > max_seeds:
            # Keep best and some diverse ones
            top = [seeds[0]]
            for s in seeds[1:]:
                if len(top) >= max_seeds:
                    break
                # Check diversity: distance from all top seeds
                diverse = True
                for ts in top:
                    if np.linalg.norm(s[0] - ts[0]) < 1.0:
                        diverse = False
                        break
                if diverse:
                    top.append(s)
            seeds = top

        # Phase 2: L-BFGS-B local optimization from each seed
        for sx, sf in seeds:
            if not time_ok() or calls_made >= budget:
                break
            
            # Wrap func to track calls and enforce budget
            def wrapped_func(x):
                nonlocal calls_made
                if calls_made >= budget or not time_ok():
                    return sf  # Return current best to stop optimization
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                calls_made += 1
                try_update(x_clipped, fx)
                return fx
            
            try:
                result = minimize(
                    wrapped_func, 
                    sx, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfun': 50}
                )
            except Exception:
                pass

        # Phase 3: Polish with (1+1)-ES if budget remains
        if best_x is not None and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            
            for _ in range(100):
                if not time_ok() or calls_made >= budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                try_update(y, fy)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.1, 5.0)
                else:
                    sigma = max(sigma * 0.85, 1e-5)

        return best_x, best_f
# EVOLVE-BLOCK-END
