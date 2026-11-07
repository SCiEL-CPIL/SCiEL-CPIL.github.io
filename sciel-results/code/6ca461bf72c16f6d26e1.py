"""Rewritten optimizer: Sobol global exploration + multi-start L-BFGS-B local refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Sobol quasi-random global search + multi-start L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 310.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # --- Global exploration: Sobol quasi-random sequence ---
        # Reserve ~60% of budget for global phase, rest for local refinement
        n_global = max(10, int(budget * 0.6))
        n_global = min(n_global, budget - 5)

        # Generate Sobol points deterministically (no RNG variance)
        sampler = qmc.Sobol(d=dim, scramble=False)
        n_pts = n_global
        # Use enough points for good coverage; Sobol is deterministic
        sobol_unit = sampler.random(n_pts)
        # Map from [0,1]^d to [-5,5]^d
        sobol_points = sobol_unit * 10.0 - 5.0
        sobol_points = np.clip(sobol_points, -5.0, 5.0)

        # Evaluate all global points
        best_x = None
        best_f = np.inf
        eval_count = 0

        for i in range(n_pts):
            if time.time() > deadline:
                break
            x = sobol_points[i]
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # --- Local refinement: multi-start L-BFGS-B ---
        # Use remaining budget for local optimization
        remaining = budget - eval_count
        if remaining <= 0 or best_x is None:
            return best_x, best_f

        # Select top candidates from global phase for local refinement
        # We'll do local refinement starting from the best global point
        # and a few nearby promising points

        # Determine how many local refinement starts we can afford
        # L-BFGS-B typically uses ~dim*2 to dim*5 evaluations per start
        # Be conservative: ~20 evals per start on average
        evals_per_start = max(10, int(remaining * 0.3 / dim))  # scale with dim
        n_starts = max(1, int(remaining / max(evals_per_start, 5)))
        n_starts = min(n_starts, 15)  # cap to avoid too many

        # Build a list of starting points: best global + a few diverse ones
        start_points = [best_x.copy()]

        # Add a few other promising global points (top-k by f-value)
        # Re-evaluate is not needed; we have the points, just pick diverse ones
        # Use points that are far from best_x for diversity
        if n_pts > 1:
            dists = np.linalg.norm(sobol_points - best_x, axis=1)
            # Pick a few points with moderate distance (not too far, not identical)
            sorted_idx = np.argsort(dists)
            # Take a few from different distance ranges
            for frac in [0.1, 0.3, 0.5, 0.7]:
                idx = int(frac * (n_pts - 1))
                idx = min(idx, n_pts - 1)
                pt = sobol_points[idx]
                # Only add if it's reasonably good (not the worst)
                # We don't have all f-values stored, so just add diverse points
                if len(start_points) < n_starts:
                    start_points.append(pt)

        # Deduplicate and limit
        seen = []
        unique_starts = []
        for sp in start_points:
            key = tuple(np.round(sp, 3))
            if key not in seen:
                seen.append(key)
                unique_starts.append(sp)
        start_points = unique_starts[:n_starts]

        # Run L-BFGS-B from each start point
        bounds = [(-5.0, 5.0)] * dim

        for sp in start_points:
            if time.time() > deadline:
                break
            if eval_count >= budget - 2:
                break

            try:
                res = minimize(
                    func,
                    sp,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxls': 50, 'maxcor': 10}
                )
                eval_count += res.nfev

                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                continue

        # --- Final polish pass: high-precision L-BFGS-B on the best candidate ---
        if best_x is not None and eval_count < budget - 2 and time.time() < deadline:
            try:
                res = minimize(
                    func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'maxls': 100, 'maxcor': 20,
                             'ftol': 1e-15, 'gtol': 1e-14}
                )
                eval_count += res.nfev
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
