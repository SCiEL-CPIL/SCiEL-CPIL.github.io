"""Optimized solver for BBOB f9 (rotated Rosenbrock) using multi-start L-BFGS-B."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer with adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Wrapper that clips to box and counts calls
        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f[0]
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x_clipped.copy()
            return f

        # Multi-start strategy: focused on Rosenbrock valley structure
        rng = np.random.default_rng(42)

        starts = []

        # 1. Origin (highest priority - Rosenbrock min at origin pre-rotation)
        starts.append(np.zeros(dim))

        # 2. Near-origin points (valley structure)
        for _ in range(12):
            starts.append(rng.normal(0, 1.0, dim))

        # 3. Points along axes (rotation detection)
        for i in range(dim):
            for val in [-2.0, -1.0, 1.0, 2.0]:
                p = np.zeros(dim)
                p[i] = val
                starts.append(p)

        # 4. Diagonal points (common in rotated problems)
        for s in [-2.0, -1.0, 1.0, 2.0]:
            starts.append(np.full(dim, s))

        # 5. Random points in likely minimum region [-4, 4]^5
        for _ in range(8):
            starts.append(rng.uniform(-3.0, 3.0, dim))

        # 6. Broader random exploration
        for _ in range(5):
            starts.append(rng.uniform(-5.0, 5.0, dim))

        # Remove duplicates
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            if not any(np.allclose(s, u, atol=1e-6) for u in unique_starts):
                unique_starts.append(s)

        # Sort: prefer points closer to origin
        starts = sorted(unique_starts, key=lambda s: np.sum(s**2))

        bounds = [(-5.0, 5.0)] * dim

        overall_best_x = None
        overall_best_f = np.inf
        converged_points = []

        # Phase 1: Multi-start with adaptive iteration limits
        for idx, start in enumerate(starts):
            if time.time() > deadline - 20.0:
                break
            if call_count[0] >= budget:
                break

            remaining_budget = budget - call_count[0]
            if remaining_budget < 15:
                break

            # Allocate more iterations to early, promising starts
            if idx < 3:
                maxiter = min(200, max(50, remaining_budget // 3))
            elif idx < 10:
                maxiter = min(100, max(30, remaining_budget // 4))
            else:
                maxiter = min(50, max(15, remaining_budget // 5))

            try:
                res = minimize(
                    counted_func,
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
                )

                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
                    converged_points.append(overall_best_x.copy())

            except Exception:
                continue

        # Phase 2: High-precision polishing on best solution
        if overall_best_x is not None and call_count[0] < budget and time.time() < deadline - 10.0:
            for polish_pass in range(4):
                if call_count[0] >= budget or time.time() > deadline - 10.0:
                    break
                remaining = budget - call_count[0]
                if remaining < 10:
                    break

                try:
                    res = minimize(
                        counted_func,
                        overall_best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': min(150, max(20, remaining // 2)),
                                 'ftol': 1e-15, 'gtol': 1e-14}
                    )
                    if res.fun < overall_best_f:
                        overall_best_f = res.fun
                        overall_best_x = np.clip(res.x, -5, 5)
                    else:
                        break  # No improvement, stop polishing

                except Exception:
                    break

        # Phase 3: Try other converged points if budget remains
        if overall_best_x is not None and call_count[0] < budget and time.time() < deadline - 5.0:
            for cp in converged_points[-3:]:
                if call_count[0] >= budget or time.time() > deadline - 5.0:
                    break
                if np.allclose(cp, overall_best_x, atol=1e-4):
                    continue
                remaining = budget - call_count[0]
                if remaining < 15:
                    break
                try:
                    res = minimize(
                        counted_func,
                        cp,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': min(50, max(10, remaining // 2)),
                                 'ftol': 1e-15, 'gtol': 1e-13}
                    )
                    if res.fun < overall_best_f:
                        overall_best_f = res.fun
                        overall_best_x = np.clip(res.x, -5, 5)
                except Exception:
                    continue

        if overall_best_x is None:
            if best_x[0] is not None:
                return best_x[0], best_f[0]
            return np.zeros(dim), 0.0

        return overall_best_x, overall_best_f
