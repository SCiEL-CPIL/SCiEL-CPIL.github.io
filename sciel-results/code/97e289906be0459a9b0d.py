import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Derivative-free optimizer for Schaffers F7 using Nelder-Mead + global restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0  # Safety margin before 330s kill
        n = self.dim
        B = self.budget

        def time_ok():
            return (time.time() - start) < deadline

        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0

        def ev(x):
            nonlocal calls, best_x, best_f
            if calls >= B or not time_ok():
                return float('inf')
            calls += 1
            y = np.clip(x, -5.0, 5.0)
            f = func(y)
            if f < best_f:
                best_f = f
                best_x = y.copy()
            return f

        rng = np.random.default_rng(12345)

        # Phase 1: Coarse global search with diverse starting points
        # Sample a set of points spread across the domain
        coarse_points = [np.zeros(n)]
        # Corners of a sub-box
        for mask in range(2 ** n):
            pt = np.array([(-4.0 if (mask >> i) & 1 else 4.0) for i in range(n)])
            coarse_points.append(pt)
        # Axis-aligned points
        for i in range(n):
            for val in [-3.0, 0.0, 3.0]:
                pt = np.zeros(n)
                pt[i] = val
                coarse_points.append(pt)
        # Random points
        for _ in range(20):
            coarse_points.append(rng.uniform(-4.0, 4.0, size=n))

        for pt in coarse_points:
            if calls >= B or not time_ok():
                break
            ev(pt)

        # Phase 2: Iterative local refinement with Nelder-Mead
        # Use multiple restarts around the best point found so far
        x0 = best_x.copy()
        no_improve = 0
        max_no_improve = 15

        for restart in range(30):
            if calls >= B or not time_ok():
                break

            # Create initial simplex around x0
            # Nelder-Mead in scipy creates its own simplex, but we can guide it
            # by providing a good initial point
            
            # Set up Nelder-Mead with tight tolerances
            # maxiter controls iterations, not function calls directly
            # But we need to be careful about budget
            
            # Use a wrapper that respects our budget
            nm_calls = [0]
            
            def nm_func(x):
                if calls >= B or not time_ok():
                    return float('inf')
                return ev(x)

            # Initial simplex: x0 + small perturbations
            # scipy's Nelder-Mead builds simplex internally
            # We'll use x0 as starting point
            
            try:
                result = minimize(
                    nm_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'maxiter': 200,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

            if calls >= B or not time_ok():
                break

            # Check if we improved
            if best_f < float('inf') and best_f < 100:
                # We have a finite best, check improvement
                pass

            x0 = best_x.copy()

            # If no improvement, try a larger perturbation
            if no_improve >= max_no_improve:
                # Restart with a more distant point
                x0 = best_x + rng.standard_normal(n) * 1.5
                x0 = np.clip(x0, -5.0, 5.0)
                no_improve = 0
            else:
                no_improve += 1

        # Phase 3: Final polish with very tight tolerances
        if calls < B and time_ok():
            try:
                def nm_func2(x):
                    if calls >= B or not time_ok():
                        return float('inf')
                    return ev(x)

                result = minimize(
                    nm_func2,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'maxiter': 100,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        return best_x, best_f
