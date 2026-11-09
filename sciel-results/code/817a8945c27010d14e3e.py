"""Optimizer for BBOB f6 (Attractive Sector) using Nelder-Mead with smart restarts."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead based optimizer with multi-start and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, calls_ref, deadline, max_calls=100):
        """Run Nelder-Mead from x0, tracking calls. Returns (best_x, best_f, calls_used)."""
        n = self.dim
        calls = [calls_ref[0]]  # mutable counter
        
        def wrapper(x):
            if calls[0] >= max_calls or time.time() > deadline:
                return 1e30  # signal to stop
            calls[0] += 1
            return float(func(x))
        
        try:
            res = minimize(wrapper, x0, method='Nelder-Mead',
                          options={'maxiter': max_calls, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            fx = float(func(res.x))
            calls[0] += 1
            return res.x, fx, calls[0] - calls_ref[0]
        except Exception:
            return x0, float(func(x0)), 1

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 330s kill
        
        rng = np.random.default_rng(42)
        calls_used = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        # Phase 1: Quick global scan - f6 has attractive sector so even random points improve
        n_scan = 20
        for i in range(n_scan):
            if time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-5.0, 5.0, dim)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Nelder-Mead from best point
        remaining = budget - calls_used
        if remaining > 10 and time.time() < deadline:
            nm_calls = min(remaining - 50, 200)  # Leave some for polish
            x_nm, f_nm, used = self._nelder_mead(func, best_x, [calls_used], deadline, max_calls=nm_calls)
            calls_used += used
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm.copy()
        
        # Phase 3: Final high-precision polish with tight tolerance
        remaining = budget - calls_used
        if remaining > 5 and time.time() < deadline:
            nm_calls = min(remaining, 100)
            x_polish, f_polish, used = self._nelder_mead(func, best_x, [calls_used], deadline, max_calls=nm_calls)
            calls_used += used
            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish.copy()
        
        # Phase 4: If we have budget left, do a few more Nelder-Mead from slightly perturbed best
        remaining = budget - calls_used
        if remaining > 20 and time.time() < deadline:
            n_restarts = min(3, remaining // 30)
            for _ in range(n_restarts):
                if time.time() > deadline or calls_used >= budget:
                    break
                perturbation = 0.1 * rng.standard_normal(dim)
                x_start = np.clip(best_x + perturbation, -5.0, 5.0)
                nm_calls = min(budget - calls_used, 50)
                x_r, f_r, used = self._nelder_mead(func, x_start, [calls_used], deadline, max_calls=nm_calls)
                calls_used += used
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
