"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start Nelder-Mead with deterministic and random initial points."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            try:
                val = func(np.array(x, dtype=float))
                if not np.isfinite(val):
                    return np.inf
                return float(val)
            except Exception:
                return np.inf

        # Generate candidate starting points
        # 1. Deterministic: center and corners
        candidates = []
        # Center
        candidates.append(np.zeros(dim))
        # Corners of the box
        for mask in range(2 ** dim):
            pt = np.array([(mask >> i) & 1 for i in range(dim)], dtype=float)
            pt = pt * 10.0 - 5.0  # map to [-5, 5]
            candidates.append(pt)
        # Midpoints of edges
        for i in range(dim):
            for j in range(i + 1, dim):
                pt = np.zeros(dim)
                pt[i] = 0.0
                pt[j] = 0.0
                candidates.append(pt)

        # 2. Random points
        rng = np.random.default_rng(42)
        for _ in range(20):
            candidates.append(rng.uniform(-5, 5, dim))

        # Sort candidates: center first, then by distance from center (closer first)
        center = np.zeros(dim)
        candidates.sort(key=lambda x: np.linalg.norm(x - center))

        # Evaluate each candidate and run Nelder-Mead from the best ones
        evaluated = []
        for cand in candidates:
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            cand = np.clip(cand, -5, 5)
            fc = safe_func(cand)
            if fc < best_f:
                best_f = fc
                best_x = cand.copy()
            evaluated.append((fc, cand))

        # Sort evaluated points by function value
        evaluated.sort(key=lambda x: x[0])

        # Run Nelder-Mead from the top few starting points
        # Limit the number of restarts to manage time
        max_restarts = min(5, len(evaluated))

        for i in range(max_restarts):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            f0, x0 = evaluated[i]

            # Skip if this point was already the best
            if i == 0 and best_x is not None and np.allclose(x0, best_x):
                # Still try to polish
                pass

            # Run Nelder-Mead
            try:
                # Options to limit function evaluations
                # maxiter controls iterations, but we need to track calls
                # We'll use a wrapper to count calls
                nm_calls = [0]
                original_budget_remaining = budget - call_count

                def nm_func(x):
                    nonlocal call_count
                    if call_count >= budget:
                        return np.inf
                    if time.time() > deadline:
                        return np.inf
                    call_count += 1
                    try:
                        val = func(np.array(x, dtype=float))
                        if not np.isfinite(val):
                            return np.inf
                        return float(val)
                    except Exception:
                        return np.inf

                # Nelder-Mead options
                # maxiter is the max number of iterations, not function calls
                # In 5D, each iteration uses up to 6 function evaluations
                # So maxiter ~ remaining_budget / 6
                max_iter = max(1, (budget - call_count) // 6)

                res = minimize(
                    nm_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )

                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()

            except Exception:
                pass

        # If we still have budget, do a few more random restarts
        if call_count < budget and time.time() < deadline:
            for _ in range(10):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                x0 = rng.uniform(-5, 5, dim)
                f0 = safe_func(x0)
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()

                # Quick local search
                try:
                    max_iter = max(1, (budget - call_count) // 6)
                    res = minimize(
                        safe_func,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-6,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
