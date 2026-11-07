"""Combined optimizer: sparse grid global search + adaptive local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: coarse grid for global basin identification,
    then adaptive Nelder-Mead for local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > deadline:
                raise TimeoutError()
            if calls_made >= budget:
                raise RuntimeError()

            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f if best_f != float('inf') else 0.0

            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val

        N = self.dim

        # Phase 1: Sparse grid search for global basin identification
        # Use 3 levels per dimension: -4, 0, 4 (minimizer is in [-4,4]^5)
        # 3^5 = 243 points, ~48% of budget
        grid_points = []
        if budget >= 250:
            levels = [-4.0, 0.0, 4.0]
            for combo in itertools.product(levels, repeat=N):
                grid_points.append(np.array(combo))
        else:
            grid_points = [np.zeros(N)]

        # Evaluate grid points
        grid_results = []
        try:
            for x in grid_points:
                f_val = evaluate(x)
                grid_results.append((f_val, x))
        except (TimeoutError, RuntimeError):
            pass

        # Sort grid results by fitness
        grid_results.sort(key=lambda x: x[0])

        # Select top K basins for local refinement
        K = 3
        top_basins = []
        for f_val, x in grid_results[:K]:
            top_basins.append(x)

        # If grid search used all budget, we are done
        if calls_made >= budget:
            return best_x, best_f

        # Phase 2: Local polish with Nelder-Mead from each top basin
        def nm_wrapper(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > deadline:
                raise TimeoutError()
            if calls_made >= budget:
                raise RuntimeError()

            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f if best_f != float('inf') else 0.0

            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val

        # Run Nelder-Mead from each top basin
        for start_x in top_basins:
            if calls_made >= budget or time.time() - start_time > deadline:
                break

            try:
                # Create initial simplex with small step size
                initial_simplex = [start_x.copy()]
                for i in range(N):
                    x_i = start_x.copy()
                    x_i[i] += 0.3
                    initial_simplex.append(x_i)

                res = minimize(
                    nm_wrapper,
                    start_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'initial_simplex': initial_simplex,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x)

            except (TimeoutError, RuntimeError):
                break
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
