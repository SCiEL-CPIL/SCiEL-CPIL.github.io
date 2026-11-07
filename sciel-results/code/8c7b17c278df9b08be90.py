"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start L-BFGS-B with gradient-based refinement for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        
        best_x = None
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")
            
            x = np.asarray(x, dtype=float)
            fx = func(x)
            eval_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            return fx

        # Phase 1: Dense initial sampling to find good starting points
        # Use Sobol sequence for space-filling coverage
        try:
            from scipy.stats import qmc
            n_scout = 40
            sampler = qmc.Sobol(d=dim, scramble=True)
            samples = sampler.random(n=n_scout)
            samples = samples * 10.0 - 5.0
            scout_points = list(samples)
        except ImportError:
            n_scout = 40
            scout_points = np.random.uniform(-5, 5, size=(n_scout, dim)).tolist()

        # Add strategic points for Rosenbrock (valley structure)
        strategic = [
            np.zeros(dim),
            np.ones(dim),
            -np.ones(dim),
            2.0 * np.ones(dim),
            -2.0 * np.ones(dim),
            np.array([1.0] * dim),
            np.array([-1.0] * dim),
        ]
        scout_points.extend(strategic)

        for p in scout_points:
            if time.time() > deadline or eval_count >= budget:
                break
            try:
                fp = wrapped_func(p)
            except Exception:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
                eval_count += 1
            except Exception:
                pass

        # Phase 2: Multiple L-BFGS-B runs from diverse starting points
        # Rosenbrock has a narrow valley; L-BFGS-B with FD gradients can follow it
        remaining = budget - eval_count
        max_starts = min(8, remaining // 40)  # Each L-BFGS-B run uses ~30-50 evals
        
        # Collect top-K starting points from initial sampling
        # We'll just use the best point and a few random perturbations
        start_points = [best_x.copy()]
        
        # Add random starts from different regions
        for _ in range(max_starts - 1):
            if time.time() > deadline or eval_count >= budget - 30:
                break
            new_start = np.random.uniform(-4, 4, size=dim)
            start_points.append(new_start)
        
        for start_x in start_points:
            if time.time() > deadline or eval_count >= budget - 20:
                break
            try:
                res = minimize(
                    wrapped_func,
                    start_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'maxfun': 60, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                break

        # Phase 3: Local polish with Nelder-Mead (derivative-free) for final refinement
        remaining = budget - eval_count
        if remaining > 10 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'maxfev': remaining, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 4: Final L-BFGS-B polish from the best point found
        remaining = budget - eval_count
        if remaining > 15 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 30, 'maxfun': remaining, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
