"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with adaptive perturbation refinement for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def make_wrapper(fallback_val):
            def wrapper(x):
                nonlocal calls_made
                if calls_made >= budget or not time_ok():
                    return fallback_val
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                calls_made += 1
                try_update(x_clipped, fx)
                return fx
            return wrapper

        # Phase 1: Initial exploration with diverse random points
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

        # Sort by function value and keep diverse top seeds
        seeds.sort(key=lambda s: s[1])
        max_seeds = 6
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

        # Phase 2: L-BFGS-B local optimization from each seed
        for sx, sf in seeds:
            if not time_ok() or calls_made >= budget:
                break
            wrapper = make_wrapper(sf)
            try:
                result = minimize(
                    wrapper,
                    sx,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 50}
                )
            except Exception:
                pass

        # Phase 3: Multi-scale refinement around best point
        if best_x is not None and calls_made < budget and time_ok():
            scales = [0.8, 0.4, 0.2, 0.1, 0.05]
            for scale in scales:
                if not time_ok() or calls_made >= budget:
                    break
                # Generate a few perturbation starts at this scale
                n_perturb = 3
                for p in range(n_perturb):
                    if not time_ok() or calls_made >= budget:
                        break
                    perturbation = rng.normal(0, scale, dim)
                    start_point = np.clip(best_x + perturbation, -5, 5)
                    wrapper = make_wrapper(best_f)
                    try:
                        result = minimize(
                            wrapper,
                            start_point,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * dim,
                            options={'maxiter': 80, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 40}
                        )
                    except Exception:
                        pass

        # Phase 4: Final high-precision polish from best point
        if best_x is not None and calls_made < budget and time_ok():
            wrapper = make_wrapper(best_f)
            try:
                result = minimize(
                    wrapper,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 100}
                )
            except Exception:
                pass

        # Phase 5: If we still have budget, do a few more random restarts with refinement
        if calls_made < budget and time_ok():
            n_extra = 5
            for i in range(n_extra):
                if not time_ok() or calls_made >= budget:
                    break
                # Sample from a tighter range around current best
                x = best_x + rng.uniform(-2, 2, dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                try_update(x, fx)
                
                wrapper = make_wrapper(best_f)
                try:
                    result = minimize(
                        wrapper,
                        x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 60, 'ftol': 1e-14, 'gtol': 1e-11, 'maxfun': 30}
                    )
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
