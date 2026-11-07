"""Multi-start L-BFGS-B with Sobol sampling and adaptive local search for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Multi-start L-BFGS-B with Sobol sampling for BBOB f22 (Gallagher 21 peaks)."""

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

        rng = np.random.default_rng(42)

        # Phase 1: Generate Sobol points for diverse initial sampling
        # Use a Sobol sequence for space-filling coverage
        n_sobol = 64
        sobol_engine = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_samples = sobol_engine.random(n_sobol)
        # Map from [0,1]^dim to [-5,5]^dim
        sobol_samples = sobol_samples * 10.0 - 5.0

        # Phase 2: Evaluate Sobol points, tracking best
        sobol_results = []
        for pt in sobol_samples:
            if eval_count >= budget - 100:
                break
            if time.time() > deadline:
                break
            f_val = wrapped_func(pt)
            update_best(pt, f_val)
            sobol_results.append((f_val, pt.copy()))

        # Sort by fitness
        sobol_results.sort(key=lambda x: x[0])

        # Phase 3: Multi-start L-BFGS-B from top Sobol points
        # Start from the best few Sobol points
        n_starts = min(8, len(sobol_results))
        start_points = []
        for i in range(n_starts):
            f_val, pt = sobol_results[i]
            start_points.append(pt)

        # Add perturbations around the best point at multiple scales
        best_pt = sobol_results[0][1] if sobol_results else np.zeros(dim)
        for scale in [0.5, 0.25, 0.1]:
            for _ in range(2):
                sp = best_pt + rng.normal(0, scale, dim)
                sp = np.clip(sp, -5.0, 5.0)
                start_points.append(sp)

        # Remove duplicates
        unique_starts = []
        for sp in start_points:
            is_dup = False
            for us in unique_starts:
                if np.linalg.norm(sp - us) < 0.05:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(sp)

        # Phase 4: Run L-BFGS-B from each starting point
        for sp in unique_starts:
            if eval_count >= budget - 20:
                break
            if time.time() > deadline:
                break

            try:
                result = minimize(
                    wrapped_func,
                    sp,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'maxls': 20, 'ftol': 1e-14, 'gtol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 5: Use remaining budget for local random search around best
        # Adaptive step size: start large, shrink if no improvement
        step_size = 0.1
        no_improve_count = 0
        max_no_improve = 30

        while eval_count < budget and time.time() < deadline:
            x = best_x + rng.normal(0, step_size, dim)
            x = np.clip(x, -5.0, 5.0)
            f_val = wrapped_func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    # Shrink step size
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break
                    no_improve_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
