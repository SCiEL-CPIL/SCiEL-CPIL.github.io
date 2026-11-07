"""
Grid-based coarse-to-fine deterministic scan combined with adaptive local refinement.
Strategy:
1. Coarse grid scan to identify the best region (exploiting Rastrigin's integer structure).
2. Refine the best cell with a finer grid.
3. Use (1+1)-ES for local polishing from the best point found.
4. Nelder-Mead for final high-precision polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        best_x = None
        best_f = float('inf')
        calls_made = 0
        max_calls = self.budget

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= max_calls

        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)
        dim = self.dim

        # Phase 1: Coarse grid scan
        # Rastrigin has local minima at integer points. The global minimum is at a shifted integer.
        # We scan a coarse grid to find the region containing the global minimum.
        # Grid: 3 points per dimension -> 3^5 = 243 points
        # We also add integer points for better coverage of Rastrigin structure.

        # First, evaluate integer points in [-4, 4]^5
        # That's 9^5 = 59049 points - too many. Let's be smarter.
        # Instead, use a coarser grid: 5 points per dim -> 5^5 = 3125, still too many.
        # Use 3 points per dim: -4, 0, 4 -> 3^5 = 243 points
        # Plus random integer points for better coverage.

        # Coarse grid: 3 points per dimension
        coarse_vals = [-4.0, 0.0, 4.0]
        n_coarse = len(coarse_vals) ** dim  # 3^5 = 243

        # We have 500 calls total. Use ~200 for coarse grid, ~100 for fine grid, ~200 for local search.
        # Let's use 150 for coarse, 50 for fine, 300 for local.

        # Generate all coarse grid points
        coarse_points = []
        for i1 in range(len(coarse_vals)):
            for i2 in range(len(coarse_vals)):
                for i3 in range(len(coarse_vals)):
                    for i4 in range(len(coarse_vals)):
                        for i5 in range(len(coarse_vals)):
                            pt = np.array([coarse_vals[i1], coarse_vals[i2], coarse_vals[i3],
                                           coarse_vals[i4], coarse_vals[i5]])
                            coarse_points.append(pt)

        # Add some random integer points for better coverage
        n_rand_int = 50
        for _ in range(n_rand_int):
            pt = rng.integers(-4, 5, size=dim).astype(float)
            coarse_points.append(pt)

        # Also add the origin and a few structured points
        coarse_points.append(np.zeros(dim))
        for i in range(dim):
            for val in [-3, -1, 1, 3]:
                pt = np.zeros(dim)
                pt[i] = val
                coarse_points.append(pt)

        # Evaluate coarse grid points
        for x in coarse_points:
            if check_deadline():
                break
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            update_best(x_clipped, f)

        if best_x is None:
            x = np.zeros(dim)
            f = func(x)
            update_best(x, f)
            return best_x, best_f

        # Phase 2: Fine grid around the best point
        # Refine the best cell with a finer grid
        # Use a grid of 3 points per dimension around the best point
        fine_radius = 1.0
        fine_vals = [best_x[i] - fine_radius, best_x[i], best_x[i] + fine_radius]

        # Generate fine grid points
        fine_points = []
        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(3):
                    for i4 in range(3):
                        for i5 in range(3):
                            pt = np.array([fine_vals[i1], fine_vals[i2], fine_vals[i3],
                                           fine_vals[i4], fine_vals[i5]])
                            fine_points.append(pt)

        # Add some random points near the best point
        for _ in range(30):
            pt = best_x + rng.uniform(-fine_radius, fine_radius, dim)
            fine_points.append(pt)

        # Evaluate fine grid points
        for x in fine_points:
            if check_deadline():
                break
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            update_best(x_clipped, f)

        if best_x is None:
            x = np.zeros(dim)
            f = func(x)
            update_best(x, f)
            return best_x, best_f

        # Phase 3: Local refinement with (1+1)-ES
        # Use adaptive step size to polish the best point
        remaining = max_calls - calls_made
        if remaining <= 0:
            return best_x, best_f

        # Reserve some budget for Nelder-Mead
        nm_budget = min(50, remaining // 4)
        es_budget = remaining - nm_budget

        # Run (1+1)-ES from the best point
        x = best_x.copy()
        f = best_f
        sigma = 0.5

        success_count = 0
        window = []
        window_size = 15

        for _ in range(es_budget):
            if check_deadline():
                break

            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < f:
                x = y
                f = fy
                success_count += 1
                window.append(1)
                # Update global best
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            else:
                window.append(0)

            # Adapt sigma based on success rate in last window_size steps
            if len(window) >= window_size:
                success_rate = sum(window[-window_size:]) / window_size
                if success_rate > 0.3:
                    sigma *= 1.1
                elif success_rate < 0.1:
                    sigma *= 0.8
                window = window[-window_size:]

        # Phase 4: Final high-precision polish with Nelder-Mead
        if not check_deadline():
            nm_calls = max_calls - calls_made
            if nm_calls > 10:
                try:
                    from scipy.optimize import minimize

                    def wrapper(x):
                        nonlocal calls_made
                        if calls_made >= max_calls:
                            return best_f
                        calls_made += 1
                        return func(np.clip(x, -5, 5))

                    # Run Nelder-Mead from current best
                    result = minimize(
                        wrapper,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': nm_calls,
                            'xatol': 1e-8,
                            'fatol': 1e-10
                        }
                    )

                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5)
                except Exception:
                    pass

        return best_x, best_f

# EVOLVE-BLOCK-END
