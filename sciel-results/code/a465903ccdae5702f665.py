"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: Random exploration + SLSQP local refinement."""

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
        n_explore = 8
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
        
        # Keep top diverse seeds (at most 4)
        max_seeds = 4
        if len(seeds) > max_seeds:
            top = [seeds[0]]
            for s in seeds[1:]:
                if len(top) >= max_seeds:
                    break
                diverse = True
                for ts in top:
                    if np.linalg.norm(s[0] - ts[0]) < 1.5:
                        diverse = False
                        break
                if diverse:
                    top.append(s)
            seeds = top

        # Phase 2: SLSQP local optimization from each seed
        for sx, sf in seeds:
            if not time_ok() or calls_made >= budget:
                break
            
            # Wrap func to track calls and enforce budget
            def wrapped_func(x):
                nonlocal calls_made
                if calls_made >= budget or not time_ok():
                    return sf
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                calls_made += 1
                try_update(x_clipped, fx)
                return fx
            
            try:
                result = minimize(
                    wrapped_func, 
                    sx, 
                    method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'ftol': 1e-12, 'maxfun': 80}
                )
            except Exception:
                pass

        # Phase 3: Final high-precision SLSQP polish from best point
        if best_x is not None and calls_made < budget:
            def polish_func(x):
                nonlocal calls_made
                if calls_made >= budget or not time_ok():
                    return best_f
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                calls_made += 1
                try_update(x_clipped, fx)
                return fx
            
            try:
                result = minimize(
                    polish_func,
                    best_x.copy(),
                    method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 500, 'ftol': 1e-15, 'maxfun': 100}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
