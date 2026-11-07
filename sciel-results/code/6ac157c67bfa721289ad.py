import numpy as np
import time


class Optimizer:
    """Katsuura-specific optimizer exploiting separable periodic structure.

    Strategy:
      1. Coarse per-coordinate grid to locate the minimum phase of each dimension.
      2. Fine per-coordinate grid to refine each coordinate.
      3. Local refinement (Nelder-Mead) for cross-coordinate coupling.
      4. Final coordinate-wise polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(7)

        best_x = np.zeros(d)
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() > deadline

        def try_point(x):
            nonlocal best_x, best_f, used_calls
            if used_calls >= budget or check_time():
                return
            x = np.asarray(x, dtype=float)
            fx = func(x)
            used_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # --- Phase 1: Coarse per-coordinate search ---
        # For each coordinate, sweep a grid while holding others at current best.
        # Katsuura has strong per-coordinate structure: each coordinate has a
        # "good phase" where the oscillatory terms align to a minimum.
        # We use a coarse grid over [-4.5, 4.5] with ~16 points per coordinate.

        # Start from origin
        try_point(np.zeros(d))

        # Coarse grid: 16 points over [-4.5, 4.5]
        coarse_grid = np.linspace(-4.5, 4.5, 16)

        # Iterate: for each coordinate, sweep the grid holding others fixed.
        # Do 2 passes to allow cross-coordinate influence.
        for _pass in range(2):
            for i in range(d):
                if used_calls >= budget or check_time():
                    return best_x, best_f
                for val in coarse_grid:
                    if used_calls >= budget or check_time():
                        break
                    x = best_x.copy()
                    x[i] = val
                    try_point(x)

        # --- Phase 2: Fine per-coordinate grid ---
        # Refine each coordinate with a finer grid around the current best.
        fine_grid = np.linspace(-4.5, 4.5, 40)

        for i in range(d):
            if used_calls >= budget or check_time():
                break
            for val in fine_grid:
                if used_calls >= budget or check_time():
                    break
                x = best_x.copy()
                x[i] = val
                try_point(x)

        # --- Phase 3: Local refinement with Nelder-Mead ---
        if best_x is not None and used_calls < budget and not check_time():
            calls_made = [0]
            max_local_calls = budget - used_calls

            def wrapped_func(x):
                calls_made[0] += 1
                if calls_made[0] > max_local_calls or check_time():
                    return 1e10
                return func(x)

            x0 = best_x.copy()
            # Build initial simplex: perturb each coordinate by a small amount
            simplex = [x0.copy()]
            for i in range(d):
                x_plus = x0.copy()
                x_plus[i] += 0.3
                simplex.append(np.clip(x_plus, -5, 5))

            try:
                from scipy.optimize import minimize
                res = minimize(wrapped_func, x0, method='Nelder-Mead',
                               options={'xatol': 1e-10, 'fatol': 1e-12,
                                        'maxiter': max_local_calls,
                                        'adaptive': True})
                used_calls += calls_made[0]
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # --- Phase 4: Final coordinate-wise polish ---
        if best_x is not None and used_calls < budget and not check_time():
            # Very fine grid around the current best for each coordinate
            for i in range(d):
                if used_calls >= budget or check_time():
                    break
                center = best_x[i]
                # Fine grid: 20 points in a window of +/- 0.1 around center
                local_grid = np.linspace(center - 0.1, center + 0.1, 20)
                for val in local_grid:
                    if used_calls >= budget or check_time():
                        break
                    x = best_x.copy()
                    x[i] = val
                    try_point(x)

            # Second polish pass with even finer grid
            for i in range(d):
                if used_calls >= budget or check_time():
                    break
                center = best_x[i]
                local_grid = np.linspace(center - 0.01, center + 0.01, 15)
                for val in local_grid:
                    if used_calls >= budget or check_time():
                        break
                    x = best_x.copy()
                    x[i] = val
                    try_point(x)

        return best_x, best_f
