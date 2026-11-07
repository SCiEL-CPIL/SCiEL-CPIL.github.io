"""Hybrid DE with Zoom-in and Nelder-Mead polishing for BBOB f4."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    """Hybrid optimizer: DE with adaptive bounds and NM polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # Track best result
        x_best = np.zeros(self.dim)
        f_best = np.inf
        calls = 0
        max_calls = self.budget

        def safe_func(x):
            nonlocal calls
            if calls >= max_calls:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            return f

        # Phase 1: Global DE on full box
        # Use a moderate population size to balance exploration and budget
        popsize = 10
        # Estimate max iterations: budget / (popsize * dim * factor)
        # DE evaluates popsize individuals per iteration roughly.
        # Let's allocate 60% of budget to Phase 1.
        budget_phase1 = int(0.6 * max_calls)
        max_iter1 = max(1, budget_phase1 // (popsize * self.dim)) # Rough estimate
        
        # SciPy DE uses 'polish' which runs NM. We disable it to control budget manually.
        try:
            res1 = differential_evolution(
                safe_func, 
                bounds=[(-5, 5)] * self.dim,
                maxiter=max_iter1,
                popsize=popsize,
                polish=False,
                seed=42,
                tol=1e-6
            )
            if res1.fun < f_best:
                f_best = res1.fun
                x_best = res1.x
        except Exception:
            pass

        # Phase 2: Local DE with zoomed-in bounds
        if calls < max_calls and time.time() < deadline:
            # Shrink bounds around best found
            radius = 1.5
            bounds2 = []
            for i in range(self.dim):
                lower = max(-5.0, x_best[i] - radius)
                upper = min(5.0, x_best[i] + radius)
                bounds2.append((lower, upper))
            
            budget_phase2 = int(0.3 * max_calls)
            popsize2 = 5
            max_iter2 = max(1, budget_phase2 // (popsize2 * self.dim))
            
            try:
                res2 = differential_evolution(
                    safe_func, 
                    bounds=bounds2,
                    maxiter=max_iter2,
                    popsize=popsize2,
                    polish=False,
                    seed=43,
                    tol=1e-6
                )
                if res2.fun < f_best:
                    f_best = res2.fun
                    x_best = res2.x
            except Exception:
                pass

        # Phase 3: Nelder-Mead Polishing
        if calls < max_calls and time.time() < deadline:
            try:
                res3 = minimize(
                    safe_func, 
                    x_best, 
                    method='Nelder-Mead',
                    options={'maxiter': max_calls - calls, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res3.fun < f_best:
                    f_best = res3.fun
                    x_best = res3.x
            except Exception:
                pass

        # Final check: if we have budget left, do some random local search
        while calls < max_calls and time.time() < deadline:
            # Small perturbations
            sigma = 0.1
            x_new = x_best + sigma * rng.standard_normal(self.dim)
            x_new = np.clip(x_new, -5, 5)
            f_new = safe_func(x_new)
            if f_new < f_best:
                f_best = f_new
                x_best = x_new

        return x_best, f_best
# EVOLVE-BLOCK-END
