"""Optimized solver for BBOB f9 (rotated Rosenbrock) using L-BFGS-B + BFGS polish."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Hybrid L-BFGS-B and BFGS optimizer for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        budget = self.budget

        # State tracking
        call_count = [0]
        best_f = [np.inf]
        best_x = [None]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f[0]
            # Ensure we pass a finite vector of correct length
            x_arr = np.asarray(x, dtype=float)
            if x_arr.shape != (dim,):
                x_arr = np.resize(x_arr, (dim,))
            x_clipped = np.clip(x_arr, -5.0, 5.0)
            f = func(x_clipped)
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x_clipped.copy()
            return f

        def remaining_time():
            return deadline - time.time()

        def has_budget():
            return call_count[0] < budget and remaining_time() > 3.0

        bounds = [(-5.0, 5.0)] * dim
        rng = np.random.default_rng(42)

        # Phase 1: Generate diverse starting points
        # The global minimum of standard Rosenbrock is (1,1,...,1).
        # With rotation and shift, it's unknown, but (1,1,...,1) and (0,0,...,0) are good anchors.
        starts = []
        starts.append(np.zeros(dim))
        starts.append(np.ones(dim))
        starts.append(np.full(dim, 0.5))
        starts.append(np.full(dim, 1.5))
        
        # Random points in the likely region [-4, 4]
        for _ in range(15):
            starts.append(rng.uniform(-4.0, 4.0, dim))
            
        # Some points on the "valley" structure of standard Rosenbrock, just in case
        for _ in range(5):
            p = rng.uniform(-2, 2, dim)
            # Perturb towards 1
            p = 0.5 * p + 0.5 * np.ones(dim)
            starts.append(p)

        # Deduplicate and clip
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            if not any(np.linalg.norm(s - u) < 0.1 for u in unique_starts):
                unique_starts.append(s)

        # Evaluate a subset of starts to find promising basins
        # We have 500 calls. Let's use 20 for initial screening.
        max_screening = min(20, len(unique_starts), budget - 100)
        screening_results = []
        for s in unique_starts[:max_screening]:
            if not has_budget():
                break
            f = counted_func(s)
            screening_results.append((f, s.copy()))
        
        screening_results.sort(key=lambda t: t[0])
        
        # Select top 5 for deep optimization
        top_starts = [s for _, s in screening_results[:5]]
        
        current_best_f = best_f[0]
        current_best_x = best_x[0].copy() if best_x[0] is not None else np.zeros(dim)

        # Phase 2: L-BFGS-B from top starts
        # L-BFGS-B is robust and handles the box constraints.
        for i, start in enumerate(top_starts):
            if not has_budget():
                break
            if remaining_time() < 20.0:
                break
            
            try:
                # Tight tolerances for the first one, slightly looser for others
                if i == 0:
                    maxiter = 200
                    ftol = 1e-14
                    gtol = 1e-12
                else:
                    maxiter = 100
                    ftol = 1e-12
                    gtol = 1e-10

                res = minimize(
                    counted_func,
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': maxiter, 
                        'ftol': ftol, 
                        'gtol': gtol, 
                        'maxls': 50
                    }
                )
                if res.fun < current_best_f:
                    current_best_f = res.fun
                    current_best_x = np.clip(res.x, -5, 5)
            except Exception:
                continue

        # Phase 3: BFGS Polish
        # BFGS uses a dense Hessian approximation, which converges faster (quadratic) 
        # near the minimum for small dimensions.
        # We assume we are inside the box, so we can use unconstrained BFGS 
        # but rely on the function clipping to be safe.
        if has_budget() and remaining_time() > 10.0:
            try:
                res = minimize(
                    counted_func,
                    current_best_x,
                    method='BFGS',
                    options={
                        'maxiter': 50, 
                        'gtol': 1e-10, 
                        'ftol': 1e-12,
                        'maxls': 50
                    }
                )
                if res.fun < current_best_f:
                    current_best_f = res.fun
                    current_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 4: Final L-BFGS-B Polish with extreme tightness
        # Sometimes BFGS can step out of the box or get stuck in a slightly suboptimal spot due to Hessian approximation errors.
        # L-BFGS-B with box constraints is safer for the final step.
        if has_budget() and remaining_time() > 5.0:
            try:
                res = minimize(
                    counted_func,
                    current_best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50, 
                        'ftol': 1e-15, 
                        'gtol': 1e-13, 
                        'maxls': 30
                    }
                )
                if res.fun < current_best_f:
                    current_best_f = res.fun
                    current_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Use tracked best if it's better (in case minimize returned a worse point due to numerical issues)
        if best_x[0] is not None and best_f[0] < current_best_f:
            current_best_f = best_f[0]
            current_best_x = best_x[0]

        if current_best_x is None:
            return np.zeros(dim), 0.0

        return current_best_x, current_best_f
