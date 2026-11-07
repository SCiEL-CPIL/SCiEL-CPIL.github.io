"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-restart L-BFGS-B with Nelder-Mead polish for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin within 300s limit

        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            x = np.clip(x, -5, 5)
            call_count += 1
            if call_count > self.budget:
                return best_f
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)
        dim = self.dim

        # Phase 1: Scatter — evaluate diverse points directly to get early progress
        scatter_points = [np.zeros(dim)]
        # Concentrated around origin
        for scale in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            for _ in range(4):
                scatter_points.append(rng.uniform(-scale, scale, dim))
        # Boundary points
        for _ in range(6):
            pts = []
            for d in range(dim):
                if rng.random() < 0.5:
                    pts.append(-4.5 + rng.random() * 0.5)
                else:
                    pts.append(4.0 + rng.random() * 0.5)
            scatter_points.append(np.array(pts))

        # Shuffle but keep origin first
        scatter_points = [scatter_points[0]] + scatter_points[1:]
        rng.shuffle(scatter_points[1:])

        # Evaluate scatter points (up to 40 calls or until budget exhausted)
        scatter_budget = min(40, self.budget // 4)
        for pt in scatter_points:
            if call_count >= scatter_budget or call_count >= self.budget:
                break
            if time.time() > deadline - 30:
                break
            pt = np.clip(pt, -5, 5)
            wrapped_func(pt)

        # Phase 2: Multi-restart L-BFGS-B from top scatter points
        # Collect evaluated points sorted by function value
        evaluated = []
        # Re-evaluate is wasteful; instead use best_x and nearby perturbations
        # Build a set of good starting points from scatter results
        # We track best_x; also use a few random perturbations of best_x
        
        good_starts = []
        if best_x is not None:
            good_starts.append(best_x.copy())
            # Add perturbations of best
            for scale in [0.5, 1.0, 2.0]:
                for _ in range(3):
                    pert = best_x + rng.normal(0, scale, dim)
                    good_starts.append(np.clip(pert, -5, 5))
        
        # Add some random starts
        for scale in [1.0, 2.0, 3.0]:
            for _ in range(3):
                good_starts.append(rng.uniform(-scale, scale, dim))

        # Shuffle
        rng.shuffle(good_starts[1:]) if len(good_starts) > 1 else None

        restarts_done = 0
        max_restarts = 12

        for start_x in good_starts:
            if call_count >= self.budget - 50:  # reserve budget for final polish
                break
            if time.time() > deadline - 40:
                break
            if restarts_done >= max_restarts:
                break

            start_x = np.clip(start_x, -5, 5)
            restarts_done += 1

            # L-BFGS-B with limited iterations
            try:
                res = minimize(
                    wrapped_func,
                    start_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-6}
                )
            except Exception:
                pass

            # Quick Nelder-Mead polish from best so far
            if best_x is not None and call_count < self.budget - 30 and time.time() < deadline - 35:
                try:
                    nm_start = best_x.copy()
                    res_nm = minimize(
                        wrapped_func,
                        nm_start,
                        method='Nelder-Mead',
                        options={'maxiter': 20, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                except Exception:
                    pass

        # Phase 3: Final high-precision polish
        if best_x is not None and call_count < self.budget - 5 and time.time() < deadline - 10:
            # Multiple rounds of tightening
            for polish_round in range(3):
                if call_count >= self.budget - 3 or time.time() > deadline - 8:
                    break
                try:
                    # Nelder-Mead with very tight tolerances
                    res_final = minimize(
                        wrapped_func,
                        best_x.copy(),
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-12, 'fatol': 1e-14}
                    )
                except Exception:
                    pass

                if call_count >= self.budget - 3 or time.time() > deadline - 8:
                    break
                try:
                    # L-BFGS-B with tight tolerances for final refinement
                    res_lbfgs = minimize(
                        wrapped_func,
                        best_x.copy(),
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                except Exception:
                    pass

        # Return the best result
        if best_x is not None:
            return best_x, best_f
        else:
            # Fallback
            z = np.zeros(dim)
            wrapped_func(z)
            return z, best_f
# EVOLVE-BLOCK-END
