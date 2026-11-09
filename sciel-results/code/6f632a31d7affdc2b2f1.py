"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    """Halton multi-start with Nelder-Mead for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        d = self.dim
        budget = self.budget

        # Wrap func to track calls
        call_count = [0]
        max_calls = budget

        def tracked_func(x):
            if call_count[0] >= max_calls:
                return np.inf
            x = np.asarray(x, dtype=float)
            if x.shape != (d,):
                x = x.reshape(d)
            result = func(x)
            call_count[0] += 1
            return result

        best_x = None
        best_f = np.inf
        best_points = [] # List of (f, x) tuples for top candidates

        def update_best(f, x):
            nonlocal best_f, best_x
            if f < best_f:
                best_f = f
                best_x = x.copy()
            # Keep a small list of top candidates for multi-start
            best_points.append((f, x.copy()))
            best_points.sort(key=lambda item: item[0])
            if len(best_points) > 10:
                best_points = best_points[:10]

        # 1. Initial Phase: Halton Sequence Sampling
        # Halton is a low-discrepancy sequence, good for covering the space uniformly.
        # We use 80 points as suggested by cell findings.
        n_init = 80
        if n_init > budget:
            n_init = budget
            
        # Halton requires prime bases. For d=5, we use the first 5 primes: 2, 3, 5, 7, 11
        # scipy.stats.qmc.Halton handles this.
        try:
            halton = qmc.Halton(d=d, scramble=True, seed=42)
            halton_points = halton.random(n_init)
        except Exception:
            # Fallback to Sobol if Halton fails
            sobol = qmc.Sobol(d=d, scramble=True, seed=42)
            halton_points = sobol.random(n_init)

        # Map from [0, 1] to [-5, 5]
        halton_points = halton_points * 10.0 - 5.0

        for i in range(n_init):
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.clip(halton_points[i], -5, 5)
            f = tracked_func(x)
            if np.isfinite(f):
                update_best(f, x)

        if best_x is None:
            # Fallback if no finite value found (unlikely)
            x = np.zeros(d)
            f = tracked_func(x)
            update_best(f, x)

        # 2. Refinement Phase: Nelder-Mead
        # Nelder-Mead is robust for sharp ridges.
        # We will run Nelder-Mead from the best point found so far.
        # We might also try from the second best if budget allows.
        
        bounds = [(-5.0, 5.0)] * d
        # Nelder-Mead doesn't support bounds directly in scipy, so we rely on the function clipping or penalty.
        # However, the problem statement says func clips to the box. So we can let NM explore outside, 
        # but the evaluator clips. This is safe.
        
        # Strategy: Run NM from the best point.
        # If we have budget left, run NM from the 2nd and 3rd best distinct points.
        
        candidates_to_refine = []
        # Add the best point
        candidates_to_refine.append(best_x)
        
        # Add other good points if they are sufficiently different
        for f, x in best_points[1:]:
            if len(candidates_to_refine) >= 3:
                break
            # Check distance to ensure diversity
            is_distinct = True
            for cx in candidates_to_refine:
                if np.linalg.norm(x - cx) < 0.5:
                    is_distinct = False
                    break
            if is_distinct:
                candidates_to_refine.append(x)

        for idx, start_x in enumerate(candidates_to_refine):
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break
            
            # Limit iterations based on remaining budget
            # NM uses ~ (d+1) * iter function evaluations per iteration roughly.
            # Let's allow a reasonable number of iterations.
            max_iter = 200
            if idx > 0:
                max_iter = 100 # Less effort for secondary starts

            try:
                # Initial simplex size
                # For sharp ridge, we might want a smaller initial step to avoid overshooting?
                # Default is 0.05 * x + 0.00025
                # Let's use a modest initial step.
                res = minimize(
                    tracked_func,
                    start_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                # The result fun might be from the last evaluation which could be outside bounds 
                # if NM stepped out, but func clips so it's fine.
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    update_best(best_f, best_x)
                else:
                    # Even if not better than global best, it might be a local min.
                    # But for AOCC, we only care about the global best trajectory.
                    pass
                    
            except Exception:
                pass

        # 3. Final Polish: If we still have budget, do a coordinate-wise or small perturbation search
        # Or just run NM again from the very best point with tighter tolerances?
        # NM with adaptive=True usually converges well.
        
        # Let's try one more NM run from the absolute best if we have budget > 50 calls left
        remaining_budget = budget - call_count[0]
        if remaining_budget > 50 and time.time() < deadline:
            try:
                res = minimize(
                    tracked_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
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
