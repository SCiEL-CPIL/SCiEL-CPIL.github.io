"""Direct solver for BBOB f9 (rotated Rosenbrock) using smart multi-start and high-precision polish."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with Rosenbrock-aware start points and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 270.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                # If budget is exceeded, return the current best to signal no improvement
                # This helps scipy stop or converge to the current best
                return best_f[0]
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x_clipped.copy()
            return f

        rng = np.random.default_rng(123)

        # Phase 1: Smart Exploration
        # Rosenbrock's minimum is at the origin in its native coordinates.
        # With rotation, the valley is a 1D subspace. Diagonal points are good candidates
        # for being close to this subspace.
        quick_starts = []
        # Origin
        quick_starts.append(np.zeros(dim))
        
        # Diagonals (highly likely to intersect the valley)
        for s in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
            quick_starts.append(np.full(dim, s))
            
        # Axis-aligned points (to catch cases where rotation aligns axes)
        for i in range(dim):
            for s in [-2.0, 2.0]:
                p = np.zeros(dim)
                p[i] = s
                quick_starts.append(p)
                
        # Random points for diversity
        for _ in range(10):
            quick_starts.append(rng.uniform(-4.0, 4.0, dim))

        # Evaluate quick starts
        start_values = []
        for x0 in quick_starts:
            if call_count[0] >= budget - 50:  # Leave budget for optimization
                break
            if time.time() > deadline - 10.0:
                break
            x0_clipped = np.clip(x0, -5.0, 5.0)
            f0 = counted_func(x0_clipped)
            start_values.append((f0, x0_clipped))

        # If no starts were evaluated (should not happen), fallback
        if not start_values:
            x0 = np.zeros(dim)
            counted_func(x0)
            start_values = [(best_f[0], x0)]

        # Sort by function value
        start_values.sort(key=lambda item: item[0])
        
        # Phase 2: Local Optimization from top starts
        # We pick the top 3 distinct starting points to run L-BFGS-B.
        # This avoids wasting budget on redundant local searches from very similar points.
        top_starts = []
        used_points = []
        for f_val, x_val in start_values:
            # Check if this point is close to any already used point
            is_unique = True
            for u_x in used_points:
                if np.linalg.norm(x_val - u_x) < 0.1:
                    is_unique = False
                    break
            if is_unique:
                top_starts.append(x_val)
                used_points.append(x_val)
            if len(top_starts) >= 3:
                break

        # If we didn't get 3 unique starts, just use the best one
        if not top_starts:
            top_starts = [start_values[0][1]]

        for x0 in top_starts:
            if call_count[0] >= budget - 20:
                break
            if time.time() > deadline - 15.0:
                break

            # Run L-BFGS-B from this start
            # maxiter is set to a reasonable number; L-BFGS-B is efficient
            try:
                res = minimize(
                    counted_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                # Update best if improved
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 3: High-Precision Polish
        # Use the best point found so far and run a very tight L-BFGS-B
        if best_x[0] is not None and call_count[0] < budget - 5 and time.time() < deadline - 5.0:
            try:
                res = minimize(
                    counted_func,
                    best_x[0],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-14}
                )
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 4: Final Check with BFGS (Dense Hessian approximation) for final nudge
        if best_x[0] is not None and call_count[0] < budget - 2 and time.time() < deadline - 2.0:
            try:
                res = minimize(
                    counted_func,
                    best_x[0],
                    method='BFGS',
                    bounds=bounds,
                    options={'maxiter': 20, 'gtol': 1e-12, 'disp': False}
                )
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Ensure we have a valid result
        if best_x[0] is None:
            best_x[0] = np.zeros(dim)
            best_f[0] = counted_func(best_x[0])

        return best_x[0], best_f[0]
