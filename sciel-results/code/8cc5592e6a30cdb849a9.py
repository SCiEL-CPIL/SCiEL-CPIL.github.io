"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """
    Multi-start L-BFGS-B optimizer for BBOB f13 (Sharp Ridge).
    
    Strategy:
    1. Use a large number of Sobol points (200) to ensure diverse coverage 
       and a high probability of landing near the sharp ridge.
    2. Identify the top 3 best initial points.
    3. Run L-BFGS-B refinement from each of these points sequentially, 
       as long as budget and time allow. This handles the sensitivity 
       of sharp ridges to initial conditions better than a single start.
    """

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

        def tracked_func(x):
            x = np.asarray(x, dtype=float)
            if x.shape != (d,):
                x = x.reshape(d)
            result = func(x)
            call_count[0] += 1
            return result

        best_x = None
        best_f = np.inf
        # Store top candidates for multi-start refinement
        # List of tuples: (f_val, x)
        candidates = []

        # 1. Initial Exploration with Sobol Points
        # Use 200 points to maximize chance of hitting the ridge vicinity
        # Leaving ~300 budget for refinement
        n_sobol = min(200, budget - 50)
        
        if n_sobol > 0:
            sobol = qmc.Sobol(d=d, scramble=True, seed=42)
            sobol_points = sobol.random(n_sobol)
            # Map from [0,1] to [-5,5]
            sobol_points = sobol_points * 10.0 - 5.0

            # Evaluate Sobol points
            for i in range(n_sobol):
                if call_count[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                
                x = np.clip(sobol_points[i], -5.0, 5.0)
                f = tracked_func(x)
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                
                # Keep track of top 3 candidates
                candidates.append((f, x.copy()))

        if best_x is None:
            # Fallback: evaluate center
            x = np.zeros(d)
            f = tracked_func(x)
            best_x = x.copy()
            best_f = f
            candidates.append((f, x.copy()))

        # Sort candidates by function value
        candidates.sort(key=lambda item: item[0])
        
        # 2. Refinement Phase
        # We will try to refine the top 3 distinct candidates.
        # L-BFGS-B is efficient, typically using 50-100 calls in 5D if close.
        
        bounds = [(-5.0, 5.0)] * d
        refined_indices = set()

        # Iterate through top candidates
        for idx, (f_cand, x_cand) in enumerate(candidates[:3]):
            # Check budget and time
            if call_count[0] >= budget - 10: # Leave small buffer
                break
            if time.time() > deadline:
                break
            
            # Avoid refining the exact same point multiple times
            # (Though Sobol points are unique, if we added duplicates in fallback, check dist)
            is_duplicate = False
            for ref_idx in refined_indices:
                if np.allclose(x_cand, candidates[ref_idx][1], atol=1e-5):
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            refined_indices.add(idx)

            try:
                # Use remaining budget for this refinement
                remaining_calls = budget - call_count[0]
                # Cap maxfun to avoid wasting too much on one start if it stalls
                # But allow enough for convergence
                maxfun_for_this = min(remaining_calls, 150)
                
                res = minimize(
                    tracked_func,
                    x_cand,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'maxfun': maxfun_for_this,
                        'gtol': 1e-8,
                        'ftol': 1e-12
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # If optimization fails, continue to next candidate
                pass

        # 3. Final Polish
        # If we have significant budget left, do a very tight polish on the current best
        remaining = budget - call_count[0]
        if remaining > 5 and time.time() < deadline and best_x is not None:
            try:
                res = minimize(
                    tracked_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'maxfun': remaining,
                        'gtol': 1e-10,
                        'ftol': 1e-12
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
