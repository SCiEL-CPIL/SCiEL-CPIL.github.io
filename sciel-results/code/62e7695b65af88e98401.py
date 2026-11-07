"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with global exploration via random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Track best solution and value
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return best_f
            calls_made += 1
            try:
                val = func(np.asarray(x, dtype=np.float64))
                val = float(val)
                if np.isfinite(val):
                    if val < best_f:
                        best_f = val
                        best_x = np.clip(np.asarray(x, dtype=np.float64), -5, 5).copy()
                    return val
                else:
                    return 1e10
            except Exception:
                return 1e10

        rng = np.random.default_rng(42)

        # Phase 1: Smart initial sampling to find the Rosenbrock valley
        # Combine: center, stratified, and random samples
        candidates = []
        
        # First: try center and a few structured points
        structured_points = [
            np.zeros(dim),
            np.ones(dim) * 0.5,
            np.ones(dim) * -0.5,
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
        ]
        
        for sp in structured_points:
            if time.time() > deadline or calls_made >= budget:
                break
            fx = safe_func(sp)
            candidates.append((fx, sp.copy()))
        
        # Stratified sampling in the box
        n_stratified = min(15, max(5, (budget - calls_made) // 20))
        for i in range(n_stratified):
            if time.time() > deadline or calls_made >= budget:
                break
            # Use Latin Hypercube-like sampling
            x = np.zeros(dim)
            for d in range(dim):
                x[d] = (i + 0.5) / n_stratified * 10.0 - 5.0 + rng.uniform(-0.5, 0.5) * (10.0 / n_stratified)
            x = np.clip(x, -5, 5)
            fx = safe_func(x)
            candidates.append((fx, x.copy()))
        
        # Random samples to fill in
        n_random = min(10, max(3, (budget - calls_made) // 25))
        for i in range(n_random):
            if time.time() > deadline or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            candidates.append((fx, x.copy()))

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Local optimization from best candidates with adaptive budget allocation
        n_local = min(8, len(candidates), max(1, (budget - calls_made) // 40))
        
        for i in range(n_local):
            if time.time() > deadline or calls_made >= budget - 15:
                break

            fx, x0 = candidates[i]
            
            # Allocate more budget to better starting points
            if i == 0:
                maxfun_alloc = max(20, (budget - calls_made) // 3)
            elif i == 1:
                maxfun_alloc = max(15, (budget - calls_made) // 4)
            elif i == 2:
                maxfun_alloc = max(10, (budget - calls_made) // 5)
            else:
                maxfun_alloc = max(5, (budget - calls_made) // 10)
            
            try:
                result = minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 300,
                        'maxfun': maxfun_alloc,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                    }
                )

                if np.isfinite(result.fun) and result.fun < best_f:
                    best_f = float(result.fun)
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

            # Check time
            if time.time() > deadline - 60:
                break

        # Phase 3: Continue with more exploratory local optimizations
        while calls_made < budget - 20 and time.time() < deadline - 30:
            # Mix: 50% random, 50% perturbation of best
            if rng.random() < 0.5:
                x = rng.uniform(-5, 5, dim)
            else:
                if best_x is not None:
                    x = best_x + rng.normal(0, 0.5, dim)
                    x = np.clip(x, -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)
            
            fx = safe_func(x)
            
            # Quick local refinement if we have budget
            if calls_made < budget - 15 and time.time() < deadline - 25:
                try:
                    result = minimize(
                        safe_func,
                        x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={
                            'maxiter': 80,
                            'maxfun': max(8, (budget - calls_made) // 8),
                            'ftol': 1e-12,
                            'gtol': 1e-8,
                        }
                    )
                    if np.isfinite(result.fun) and result.fun < best_f:
                        best_f = float(result.fun)
                        best_x = np.clip(result.x, -5, 5).copy()
                except Exception:
                    pass

        # Phase 4: Final high-precision polish
        if best_x is not None and calls_made < budget - 5 and time.time() < deadline - 10:
            # First polish pass
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 500,
                        'maxfun': max(10, (budget - calls_made) // 2),
                        'ftol': 1e-16,
                        'gtol': 1e-14,
                    }
                )
                if np.isfinite(result.fun) and result.fun < best_f:
                    best_f = float(result.fun)
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

            # Second ultra-precise polish
            if best_x is not None and calls_made < budget - 3 and time.time() < deadline - 5:
                try:
                    result2 = minimize(
                        safe_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={
                            'maxiter': 1000,
                            'maxfun': max(5, budget - calls_made - 1),
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
            if calls_made < budget:
                best_f = safe_func(best_x)
            else:
                best_f = 0.0

        return best_x, best_f


# EVOLVE-BLOCK-END
