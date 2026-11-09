"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best solution
        best_x = None
        best_f = np.inf
        call_count = 0

        def count_calls(x):
            nonlocal call_count
            call_count += 1
            return func(x)

        # Generate diverse starting points using Halton + random
        def halton_sequence(n, dim, skip=1000):
            """Generate n points in [0,1]^dim using Halton sequence."""
            points = []
            for i in range(skip, skip + n):
                point = []
                for j in range(dim):
                    base = j + 2
                    f = 1.0
                    r = 0.0
                    i_val = i
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    point.append(r)
                points.append(point)
            return np.array(points)

        # Create diverse initial points
        n_starts = 40
        # Mix of Halton and random points
        halton_pts = halton_sequence(n_starts // 2, dim)
        random_pts = rng.uniform(0, 1, size=(n_starts - n_starts // 2, dim))
        init_pts = np.vstack([halton_pts, random_pts])
        
        # Map to [-5, 5]
        init_pts = -5 + 10 * init_pts

        # Shuffle
        perm = rng.permutation(n_starts)
        init_pts = init_pts[perm]

        bounds = [(-5, 5)] * dim

        for i in range(n_starts):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            x0 = init_pts[i]
            try:
                result = minimize(
                    count_calls,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxfun': 50}
                )
            except Exception:
                continue

            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()

            # If we found a very good solution, try more local refinement
            if best_f < 0.01 and call_count < budget:
                # Additional local search around the best point
                for _ in range(3):
                    if call_count >= budget:
                        break
                    if time.time() > deadline:
                        break
                    try:
                        result2 = minimize(
                            count_calls,
                            best_x + rng.normal(0, 0.1, dim),
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 100, 'maxfun': 30}
                        )
                        if result2.fun < best_f:
                            best_f = result2.fun
                            best_x = result2.x.copy()
                    except Exception:
                        pass

        # If we still have budget, do a final round of random restarts
        remaining = budget - call_count
        if remaining > 10 and time.time() < deadline:
            n_extra = min(remaining // 5, 20)
            for _ in range(n_extra):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                x0 = rng.uniform(-5, 5, dim)
                try:
                    result = minimize(
                        count_calls,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'maxfun': 20}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        if best_x is None:
            # Fallback: just use the first point
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
