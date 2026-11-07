"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer for BBOB-style f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 325.0  # safety margin before 330s kill

        n = self.dim
        budget = self.budget
        calls_made = 0

        # Wrap func to count calls and clip
        def wrapped(x):
            nonlocal calls_made
            x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = func(x_clipped)
            calls_made += 1
            return val

        # Multi-start strategy: spread initial points across the domain
        # Use a grid + random combination for coverage
        rng = np.random.default_rng(42)

        # Generate initial points: combination of grid, random, and center
        init_points = []

        # Center
        init_points.append(np.zeros(n))

        # Grid points (3^5 = 243 is too many, use 2^5 = 32)
        grid_vals = np.linspace(-4, 4, 2)
        for i in range(2 ** n):
            pt = np.array([grid_vals[(i >> j) & 1] for j in range(n)])
            init_points.append(pt)

        # Add random points to fill budget
        remaining = budget - len(init_points)
        if remaining > 0:
            n_random = min(remaining, 50)
            random_pts = rng.uniform(-5, 5, size=(n_random, n))
            init_points.extend(random_pts.tolist())

        # Sort by distance to center (prefer exploring near center first, then outward)
        # Actually for multimodal, let's interleave: start with diverse points
        # Shuffle to avoid bias
        rng.shuffle(init_points)

        # Also add some structured points
        # Points along each axis
        for axis in range(n):
            for val in [-4, -2, 0, 2, 4]:
                pt = np.zeros(n)
                pt[axis] = val
                init_points.append(pt)

        # Remove duplicates approximately
        unique_points = []
        for pt in init_points:
            pt = np.array(pt)
            if all(np.linalg.norm(pt - u) > 0.1 for u in unique_points):
                unique_points.append(pt)

        # We'll use L-BFGS-B from each starting point
        # Each local optimization uses a budget of evaluations
        # Total budget: 500
        # Strategy: use ~20-30 evaluations per local run, so ~15-25 restarts

        # Adaptive budget per restart: start with smaller, increase for promising starts
        # Simpler: fixed budget per restart
        evals_per_restart = 25
        max_restarts = budget // evals_per_restart

        best_x = None
        best_f = np.inf

        # Evaluate initial points to find good starting locations
        # But we need to be careful: each evaluation counts
        # Let's just run L-BFGS-B from each starting point with a limited maxiter

        # Actually, let's do a two-phase approach:
        # Phase 1: Quick scan with 1 eval each from diverse points
        # Phase 2: L-BFGS-B from the best few starting points

        # Phase 1: Evaluate a subset of points
        scan_points = unique_points[:min(100, len(unique_points))]
        scan_results = []
        for pt in scan_points:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            x_pt = np.clip(np.array(pt), -5, 5)
            f_pt = wrapped(x_pt)
            scan_results.append((f_pt, x_pt))
            if f_pt < best_f:
                best_f = f_pt
                best_x = x_pt.copy()

        # Phase 2: L-BFGS-B from the top starting points
        scan_results.sort(key=lambda x: x[0])
        top_starts = scan_results[:min(10, len(scan_results))]

        for f_start, x_start in top_starts:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            # Run L-BFGS-B from this starting point
            try:
                result = minimize(
                    wrapped,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 50, 'maxfun': max(1, budget - calls_made - 5)}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

            if calls_made >= budget:
                break

        # If we still have budget, do more L-BFGS-B from random points
        if best_x is None:
            best_x = np.zeros(n)
            best_f = wrapped(best_x)

        # Continue with more restarts if budget allows
        while calls_made < budget - 10:
            if time.time() > deadline:
                break

            # Random restart
            x_rand = rng.uniform(-5, 5, size=n)
            f_rand = wrapped(x_rand)
            if f_rand < best_f:
                best_f = f_rand
                best_x = x_rand.copy()

            # Try L-BFGS-B from this point
            if calls_made < budget - 20:
                try:
                    result = minimize(
                        wrapped,
                        x_rand,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * n,
                        options={'maxiter': 30, 'maxfun': max(1, budget - calls_made - 5)}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        # Final refinement with L-BFGS-B from the best point
        if calls_made < budget - 5:
            try:
                result = minimize(
                    wrapped,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 100, 'maxfun': max(1, budget - calls_made)}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
