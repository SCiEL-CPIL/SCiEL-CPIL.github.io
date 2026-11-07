"""CMA-ES based optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with L-BFGS-B refinement for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Evaluation counter
        eval_count = 0

        def wrapped_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Structured grid exploration for strong early AOCC
        # Use a 3^5 grid (243 points) evaluated in shuffled order
        rng = np.random.default_rng(123)
        
        # Generate 3^5 grid points: each dimension takes values -3, 0, 3
        grid_vals = np.array([-3.0, 0.0, 3.0])
        grid_points = []
        for a in grid_vals:
            for b in grid_vals:
                for c in grid_vals:
                    for d in grid_vals:
                        for e in grid_vals:
                            grid_points.append([a, b, c, d, e])
        grid_points = np.array(grid_points)
        
        # Shuffle for random-like exploration order
        perm = rng.permutation(len(grid_points))
        shuffled_grid = grid_points[perm]
        
        # Evaluate grid points, stopping when we have enough budget left for local optimization
        # Reserve ~100 evals for local refinement
        reserve_for_local = 100
        max_grid_evals = min(len(shuffled_grid), budget - reserve_for_local)
        
        for pt in shuffled_grid[:max_grid_evals]:
            if time.time() > deadline:
                break
            f_val = wrapped_func(pt)
            update_best(pt, f_val)
            
        # Phase 2: Local refinement from top basins
        # Collect all evaluated points and their fitness
        # Re-evaluate best few points to get exact values (they were already evaluated)
        
        # Run L-BFGS-B from the best point and a few diverse points
        start_points = [best_x.copy()]
        
        # Add a few random points around the best
        for scale in [1.5, 0.8, 0.4]:
            for _ in range(2):
                sp = best_x + rng.normal(0, scale, dim)
                sp = np.clip(sp, -5.0, 5.0)
                start_points.append(sp)
        
        # Add a couple of random points from the grid that were good
        # (We don't track all, so just add random)
        for _ in range(3):
            sp = rng.uniform(-4.0, 4.0, dim)
            start_points.append(sp)
        
        # Remove duplicates
        unique_starts = []
        for sp in start_points:
            is_dup = False
            for us in unique_starts:
                if np.linalg.norm(sp - us) < 0.1:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(sp)
        
        # Run L-BFGS-B from each starting point with tight budget
        for sp in unique_starts:
            if eval_count >= budget - 5:
                break
            if time.time() > deadline:
                break
            
            try:
                result = minimize(
                    wrapped_func,
                    sp,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 20, 'maxls': 10, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # Phase 3: Final high-precision polish
        if eval_count < budget - 3 and time.time() < deadline:
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # Fill remaining budget with fine-grained search around best
        while eval_count < budget and time.time() < deadline:
            x = best_x + rng.normal(0, 0.05, dim)
            x = np.clip(x, -5.0, 5.0)
            f_val = wrapped_func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
