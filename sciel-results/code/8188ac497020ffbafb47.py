"""Surrogate-assisted derivative-free optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-stage derivative-free optimizer:
    1. Strategic initialization (center + axis-aligned) + random sampling.
    2. Nelder-Mead local refinements from diverse starting points.
    3. (1+1)-ES polish with adaptive step size.
    
    Uses a time guard to stay within the 300s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin

        rng = np.random.default_rng(42)
        calls_used = 0
        best_x = None
        best_f = np.inf

        # Store top points from random phase for diverse NM seeds
        top_points = []

        def check_time():
            return (time.time() - start_time) > (deadline - 5.0)

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f
                # Keep a small list of top points
                top_points.append((f, x.copy()))
                top_points.sort(key=lambda item: item[0])
                if len(top_points) > 8:
                    top_points.pop()

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_used += 1
            if f < best_f:
                best_x = x.copy()
                best_f = f
                top_points.append((f, x.copy()))
                top_points.sort(key=lambda item: item[0])
                if len(top_points) > 8:
                    top_points.pop()
            return f

        # Phase 1: Strategic initialization - center and axis-aligned points
        init_points = [np.zeros(self.dim)]
        for i in range(self.dim):
            for delta in [-2.5, 2.5]:
                pt = np.zeros(self.dim)
                pt[i] = delta
                init_points.append(pt)

        for pt in init_points:
            if calls_used >= self.budget or check_time():
                break
            evaluate(pt)

        # Phase 1b: Random sampling for broad exploration
        # f11 is multimodal. ~30 points is a good balance for 5D with budget 500.
        n_random = min(30, self.budget - calls_used)
        for _ in range(n_random):
            if calls_used >= self.budget or check_time():
                break
            x = rng.uniform(-5, 5, self.dim)
            evaluate(x)

        if calls_used >= self.budget or check_time():
            return best_x, best_f

        # Phase 2: Nelder-Mead local refinements
        # NM is highly effective for the smooth parts of f11.
        # We use multiple restarts to escape local minima.

        # Determine how many NM restarts we can afford.
        # Each NM run uses ~40-60 evals.
        # Reserve ~50 evals for final ES polish.
        remaining = self.budget - calls_used - 50
        n_nm = max(1, min(8, remaining // 45))

        # Select diverse starting points
        # 1. The global best so far
        # 2. The next best distinct points from Phase 1
        # 3. Random perturbations of the best point if we need more seeds

        start_points = []

        # Always start from the best
        if best_x is not None:
            start_points.append(best_x.copy())

        # Add other top points from Phase 1 if they are distinct enough
        for f_val, x_val in top_points[1:]:
            if len(start_points) >= n_nm:
                break
            # Check distance to existing start points to ensure diversity
            is_distinct = True
            for sp in start_points:
                if np.linalg.norm(x_val - sp) < 1.0:
                    is_distinct = False
                    break
            if is_distinct:
                start_points.append(x_val.copy())

        # Fill remaining slots with perturbations of the best point
        while len(start_points) < n_nm:
            if calls_used >= self.budget or check_time():
                break
            perturbation = rng.normal(0, 1.5, self.dim)
            x_start = np.clip(best_x + perturbation, -5, 5)
            start_points.append(x_start)

        for x_start in start_points:
            if calls_used >= self.budget or check_time():
                break

            # Track local calls to prevent NM from consuming all budget
            local_calls = [0]
            max_local_calls = 45

            def nm_func(x):
                nonlocal calls_used, local_calls
                if local_calls[0] >= max_local_calls or calls_used >= self.budget:
                    return best_f  # Return current best to stop optimization gracefully
                x_clipped = np.clip(x, -5, 5)
                f = func(x_clipped)
                calls_used += 1
                local_calls[0] += 1
                if f < best_f:
                    best_x = x_clipped.copy()
                    best_f = f
                    top_points.append((f, x_clipped.copy()))
                    top_points.sort(key=lambda item: item[0])
                    if len(top_points) > 8:
                        top_points.pop()
                return f

            try:
                # Tighter tolerances for better precision
                result = minimize(
                    nm_func,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    try_update(result.x, result.fun)
            except Exception:
                pass

        if calls_used >= self.budget or check_time():
            return best_x, best_f

        # Phase 3: (1+1)-ES polish
        # Fine-tune the best solution found by NM
        x = best_x.copy()
        fx = best_f
        sigma = 0.5

        while calls_used < self.budget and not check_time():
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x, fx = y, fy
                try_update(x, fx)
                sigma *= 1.3
            else:
                sigma *= 0.85

            if sigma < 1e-6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
