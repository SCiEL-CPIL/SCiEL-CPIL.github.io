"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize as _scipy_minimize


class Optimizer:
    """Grid-based global search with local refinement for shifted/rotated
    Rosenbrock on [-5, 5]^5.  Uses a coarse lattice to find the basin, then
    Nelder-Mead + L-BFGS-B for refinement."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.perf_counter()

        # Deadline guard: return best before 330s kill
        def deadline():
            return time.perf_counter() - t0 > 260.0

        # Track calls and best
        call_count = 0
        best_x = None
        best_f = float("inf")

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return float("inf")
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                return float("inf")
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Coarse grid search over [-4, 4]^5 ---
        # Use a small number of points per dimension to stay within budget
        # 5 points per dim = 3125 points, too many. Use 3 points per dim = 243
        # Actually, let's use a stratified approach:
        # First evaluate center, then expand outward.

        # Center point
        center = np.zeros(dim)
        eval_func(center)

        # Coarse lattice: 3 levels per dimension in [-4, 4]
        # That's 3^5 = 243 points. With budget 500, we can afford this plus refinement.
        # But 243 grid + refinement might exceed 500. Let's be smarter.
        
        # Use a two-phase approach:
        # Phase 1a: Very coarse (2 levels: -3, 3) = 2^5 = 32 points
        # Phase 1b: Medium (3 levels: -3, 0, 3) around best
        # Phase 2: Local refinement

        # Phase 1a: Coarse corners
        coarse_vals = [-3.5, 0.0, 3.5]
        grid_points = []
        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(3):
                    for i4 in range(3):
                        for i5 in range(3):
                            pt = np.array([coarse_vals[i1], coarse_vals[i2],
                                           coarse_vals[i3], coarse_vals[i4],
                                           coarse_vals[i5]])
                            grid_points.append(pt)

        # Sort by distance from center to prioritize
        center_dist = np.zeros(dim)
        grid_points.sort(key=lambda p: np.sum((p - center_dist) ** 2))

        # Evaluate grid points, stopping if budget or deadline
        for pt in grid_points:
            if deadline() or call_count >= budget - 50:
                break
            eval_func(pt)

        # --- Phase 2: Local refinement from best grid point ---
        if best_x is not None and call_count < budget - 10:
            # Try L-BFGS-B from best point
            try:
                res = _scipy_minimize(
                    eval_func,
                    best_x.copy(),
                    method="L-BFGS-B",
                    bounds=[(-5.0, 5.0)] * dim,
                    options={"maxiter": 100, "maxfun": 100,
                             "gtol": 1e-10, "ftol": 1e-14},
                )
                rx = np.clip(np.asarray(res.x, dtype=float), -5.0, 5.0)
                eval_func(rx)
            except Exception:
                pass

        # Nelder-Mead polish
        if best_x is not None and not deadline() and call_count < budget - 5:
            try:
                res = _scipy_minimize(
                    eval_func,
                    best_x.copy(),
                    method="Nelder-Mead",
                    bounds=[(-5.0, 5.0)] * dim,
                    options={"maxiter": 200, "maxfev": 80,
                             "xatol": 1e-10, "fatol": 1e-14},
                )
                rx = np.clip(np.asarray(res.x, dtype=float), -5.0, 5.0)
                eval_func(rx)
            except Exception:
                pass

        # --- Phase 3: If still have budget, try additional local searches ---
        # From several promising grid points
        if not deadline() and call_count < budget - 20:
            # Re-evaluate top few grid points with local optimization
            # Find top 5 distinct regions from grid
            evaluated = []
            for pt in grid_points:
                if len(evaluated) >= 5:
                    break
                # Check if this point is near already-evaluated ones
                is_new = True
                for ev_pt in evaluated:
                    if np.linalg.norm(pt - ev_pt) < 1.0:
                        is_new = False
                        break
                if is_new:
                    evaluated.append(pt)

            for pt in evaluated:
                if deadline() or call_count >= budget - 5:
                    break
                # Local search from this point
                try:
                    res = _scipy_minimize(
                        eval_func,
                        pt.copy(),
                        method="L-BFGS-B",
                        bounds=[(-5.0, 5.0)] * dim,
                        options={"maxiter": 50, "maxfun": 50,
                                 "gtol": 1e-10, "ftol": 1e-14},
                    )
                    rx = np.clip(np.asarray(res.x, dtype=float), -5.0, 5.0)
                    eval_func(rx)
                except Exception:
                    pass

        # --- Phase 4: Final polish if budget remains ---
        if best_x is not None and not deadline() and call_count < budget - 2:
            try:
                res = _scipy_minimize(
                    eval_func,
                    best_x.copy(),
                    method="Nelder-Mead",
                    bounds=[(-5.0, 5.0)] * dim,
                    options={"maxiter": 100, "maxfev": budget - call_count - 1,
                             "xatol": 1e-12, "fatol": 1e-16},
                )
                rx = np.clip(np.asarray(res.x, dtype=float), -5.0, 5.0)
                eval_func(rx)
            except Exception:
                pass

        # Fallback
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
            except Exception:
                best_f = float("inf")

        return best_x, best_f
# EVOLVE-BLOCK-END
