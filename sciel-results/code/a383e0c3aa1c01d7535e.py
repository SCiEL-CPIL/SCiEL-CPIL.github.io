"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with smart initialization for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Track best solution and value
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return best_f
            calls_made += 1
            try:
                val = func(np.asarray(x, dtype=np.float64))
                val = float(val)
                if np.isfinite(val):
                    return val
                else:
                    return 1e10
            except Exception:
                return 1e10

        rng = np.random.default_rng(42)
        bounds = [(-5, 5)] * dim

        # Phase 1: Smart initial sampling
        # Mix of uniform, Gaussian centered at origin, and Rosenbrock-aware points
        n_random = min(30, budget // 8)
        candidates = []

        for i in range(n_random):
            if time.time() > deadline or calls_made >= budget:
                break
            if i < 10:
                # Uniform samples
                x = rng.uniform(-5, 5, dim)
            elif i < 20:
                # Gaussian samples centered at origin (sigma=2)
                x = rng.normal(0, 2, dim)
                x = np.clip(x, -5, 5)
            else:
                # Points near the typical Rosenbrock valley structure
                # For standard Rosenbrock, the valley is near x_i ≈ 1
                # With hidden shift/rotation, sample around various centers
                center = rng.uniform(-3, 3, dim)
                x = center + rng.normal(0, 1, dim)
                x = np.clip(x, -5, 5)
            
            fx = safe_func(x)
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Local optimization from promising candidates
        # Only optimize from candidates that are competitive with current best
        n_local = min(8, len(candidates), max(1, (budget - calls_made) // 40))

        for i in range(n_local):
            if time.time() > deadline or calls_made >= budget - 20:
                break

            fx, x0 = candidates[i]

            # Skip if this candidate is much worse than current best
            # (allow some slack for numerical noise)
            if fx > best_f + 1e-6:
                continue

            try:
                result = minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 300,
                        'maxfun': max(20, min(100, budget - calls_made - 30)),
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                    }
                )

                if np.isfinite(result.fun) and result.fun < best_f:
                    best_f = float(result.fun)
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 3: Adaptive refinement loop
        # Alternate between exploration and exploitation
        while calls_made < budget - 10 and time.time() < deadline - 15:
            # Explore: try a new random point
            x_new = rng.uniform(-5, 5, dim)
            fx_new = safe_func(x_new)
            
            if fx_new < best_f:
                best_f = fx_new
                best_x = x_new.copy()

            # Exploit: refine the best point found
            if best_x is not None and calls_made < budget - 15 and time.time() < deadline - 20:
                try:
                    result = minimize(
                        safe_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 100,
                            'maxfun': max(10, min(50, budget - calls_made - 10)),
                            'ftol': 1e-14,
                            'gtol': 1e-10,
                        }
                    )
                    if np.isfinite(result.fun) and result.fun < best_f:
                        best_f = float(result.fun)
                        best_x = np.clip(result.x, -5, 5).copy()
                except Exception:
                    pass

        # Phase 4: Final high-precision polish
        if best_x is not None and calls_made < budget - 5 and time.time() < deadline - 5:
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 500,
                        'maxfun': max(5, budget - calls_made - 2),
                        'ftol': 1e-16,
                        'gtol': 1e-14,
                    }
                )
                if np.isfinite(result.fun) and result.fun < best_f:
                    best_f = float(result.fun)
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

            # Ultra-high-precision final pass
            if best_x is not None and calls_made < budget - 2 and time.time() < deadline - 2:
                try:
                    result2 = minimize(
                        safe_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 1000,
                            'maxfun': max(3, budget - calls_made),
                            'ftol': 1e-18,
                            'gtol': 1e-16,
                        }
                    )
                    if np.isfinite(result2.fun) and result2.fun < best_f:
                        best_f = float(result2.fun)
                        best_x = np.clip(result2.x, -5, 5).copy()
                except Exception:
                    pass

        # Final check
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f


# EVOLVE-BLOCK-END
