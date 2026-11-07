"""Optimizer using multi-start Nelder-Mead with adaptive restarts for box-constrained minimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer for box-constrained minimization on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            call_count[0] += 1
            val = float(func(x))
            return val

        def time_ok():
            return time.time() < deadline

        def run_nelder_mead(x0, maxiter=200):
            """Run Nelder-Mead from x0, return (success, result)."""
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                return True, res
            except Exception:
                return False, None

        # Phase 1: Quick diverse initial sampling to find good starting regions
        # Evaluate structured points + random points to quickly locate promising basins
        rng = np.random.default_rng(42)
        
        # Structured grid-like points (coarse) to cover the space
        structured_starts = []
        # Center
        structured_starts.append(np.zeros(dim))
        # Axis-aligned points
        for i in range(dim):
            for sign in [-1, 1]:
                pt = np.zeros(dim)
                pt[i] = sign * 3.0
                structured_starts.append(pt)
        # Corner-ish points
        for i in range(min(8, 2**dim)):
            pt = np.zeros(dim)
            for j in range(dim):
                pt[j] = (1 if (i >> j) & 1 else -1) * 3.5
            structured_starts.append(pt)
        
        # Add random points
        n_random = 20
        for _ in range(n_random):
            if call_count[0] >= budget - 50:
                break
            structured_starts.append(rng.uniform(-4, 4, dim))
        
        # Evaluate all starting points
        candidates = []
        for x0 in structured_starts:
            if not time_ok() or call_count[0] >= budget - 30:
                break
            x0 = np.clip(x0, -5, 5)
            try:
                f0 = wrapped_func(x0)
                candidates.append((f0, x0.copy()))
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()
            except Exception:
                continue
        
        # Sort candidates by function value
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Nelder-Mead from top candidates (most promising first)
        # Use up to 6 of the best starting points
        n_polish = min(6, len(candidates))
        for i in range(n_polish):
            if not time_ok() or call_count[0] >= budget - 10:
                break
            
            f0, x0 = candidates[i]
            # If this candidate is already worse than best, skip
            if f0 > best_f * 1.5 and best_f != np.inf:
                continue
            
            success, res = run_nelder_mead(x0, maxiter=150)
            if success and res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        
        # Phase 3: Adaptive restarts near the best solution
        # If we have budget left, do local restarts around the current best
        if best_x is not None and time_ok() and call_count[0] < budget - 10:
            rng2 = np.random.default_rng(123)
            n_restarts = 5
            for i in range(n_restarts):
                if not time_ok() or call_count[0] >= budget - 5:
                    break
                
                # Start near the best solution with decreasing perturbation
                scale = 1.0 / (i + 1)
                x0 = best_x + rng2.normal(0, scale, dim)
                x0 = np.clip(x0, -5, 5)
                
                success, res = run_nelder_mead(x0, maxiter=100)
                if success and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
        
        # Phase 4: If still have budget, try a few more random restarts
        if best_x is not None and time_ok() and call_count[0] < budget - 10:
            rng3 = np.random.default_rng(456)
            n_extra = 3
            for i in range(n_extra):
                if not time_ok() or call_count[0] >= budget - 5:
                    break
                x0 = rng3.uniform(-4, 4, dim)
                success, res = run_nelder_mead(x0, maxiter=80)
                if success and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()

        # Fallback if nothing was found
        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
