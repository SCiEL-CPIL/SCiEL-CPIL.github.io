"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid global scout + multi-start L-BFGS-B + ES refinement for Rosenbrock-style f8."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s limit

        # Track best solution and function value
        best_x = None
        best_f = np.inf
        call_count = 0
        dim = self.dim
        budget = self.budget

        def wrapped(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time exceeded")
            call_count += 1
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        rng = np.random.default_rng(12345)

        # Phase 1: Global Scout
        # Strategic points: origin, standard min candidates, corners, midpoints
        scout_points = []

        # Origin and standard candidates
        scout_points.append(np.zeros(dim))
        scout_points.append(np.ones(dim))
        scout_points.append(-np.ones(dim))
        scout_points.append(np.array([0.5] * dim))
        scout_points.append(np.array([-0.5] * dim))

        # Corners (±5)
        for i in range(2 ** dim):
            p = np.zeros(dim)
            for j in range(dim):
                if (i >> j) & 1:
                    p[j] = 5.0
                else:
                    p[j] = -5.0
            scout_points.append(p)

        # Mid-points (±2.5)
        for i in range(2 ** dim):
            p = np.zeros(dim)
            for j in range(dim):
                if (i >> j) & 1:
                    p[j] = 2.5
                else:
                    p[j] = -2.5
            scout_points.append(p)

        # Add some random points for diversity
        for _ in range(10):
            scout_points.append(rng.uniform(-4.5, 4.5, dim))

        # Evaluate all scout points
        for s in scout_points:
            if call_count >= budget or time.time() > deadline:
                break
            s_clipped = np.clip(s, -5.0, 5.0)
            try:
                wrapped(s_clipped)
            except (RuntimeError, Exception):
                break

        # Phase 2: Multi-start L-BFGS-B from top scout points
        # Sort scout points by their function values and pick top candidates
        # We'll track the best few points found during scouting
        if best_x is not None:
            # Collect candidate starting points: best point and a few others
            # Use the best point and generate perturbations around it
            start_candidates = [best_x.copy()]

            # Add a few random perturbations around the best point
            for _ in range(3):
                perturbed = best_x + rng.uniform(-1.0, 1.0, dim)
                start_candidates.append(np.clip(perturbed, -5.0, 5.0))

            # Add origin and standard min as additional candidates if not already there
            start_candidates.append(np.zeros(dim))
            start_candidates.append(np.ones(dim))

            # Run L-BFGS-B from each candidate, allocating remaining budget
            remaining_budget = budget - call_count
            num_starts = min(len(start_candidates), 5)
            per_start_budget = max(1, remaining_budget // num_starts)

            for i in range(num_starts):
                if call_count >= budget or time.time() > deadline:
                    break

                x0 = start_candidates[i]
                remaining_fev = max(1, budget - call_count)

                try:
                    result = minimize(
                        wrapped,
                        x0,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={
                            'maxiter': 200,
                            'maxfev': remaining_fev,
                            'gtol': 1e-8,
                            'ftol': 1e-10,
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5.0, 5.0)
                except (RuntimeError, Exception):
                    pass

        # Phase 3: (1+1)-ES refinement to escape local minima and polish
        if best_x is not None and call_count < budget and time.time() < deadline:
            x_es = best_x.copy()
            fx_es = best_f
            sigma = 0.1
            max_es_calls = min(80, budget - call_count)

            for _ in range(max_es_calls):
                if call_count >= budget or time.time() > deadline:
                    break
                y = np.clip(x_es + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = wrapped(y)
                except (RuntimeError, Exception):
                    break
                if fy < fx_es:
                    x_es, fx_es = y, fy
                    sigma *= 1.25
                else:
                    sigma *= 0.85

                if fx_es < best_f:
                    best_f = fx_es
                    best_x = x_es.copy()

        # Phase 4: Final polish with Powell from the very best
        if best_x is not None and call_count < budget and time.time() < deadline:
            remaining_fev = max(1, budget - call_count)
            try:
                result2 = minimize(
                    wrapped,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 150,
                        'maxfev': remaining_fev,
                        'xtol': 1e-12,
                        'ftol': 1e-14,
                    }
                )
                if result2.fun < best_f:
                    best_f = result2.fun
                    best_x = np.clip(result2.x, -5.0, 5.0)
            except (RuntimeError, Exception):
                pass

        # Phase 5: Final high-precision ES polish
        if best_x is not None and call_count < budget and time.time() < deadline:
            x_es = best_x.copy()
            fx_es = best_f
            sigma = 0.02
            max_es_calls = min(40, budget - call_count)

            for _ in range(max_es_calls):
                if call_count >= budget or time.time() > deadline:
                    break
                y = np.clip(x_es + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = wrapped(y)
                except (RuntimeError, Exception):
                    break
                if fy < fx_es:
                    x_es, fx_es = y, fy
                    sigma *= 1.15
                else:
                    sigma *= 0.85

                if fx_es < best_f:
                    best_f = fx_es
                    best_x = x_es.copy()

        # Ensure we have a valid return
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
            except Exception:
                best_f = np.inf

        return best_x, best_f
# EVOLVE-BLOCK-END
