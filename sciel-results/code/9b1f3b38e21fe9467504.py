import numpy as np
import time
import scipy.optimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, use 290s to be safe
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        eval_count = 0

        def counting_func(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = float(func(x_clipped))
            eval_count += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Phase 1: Initial Exploration with Sobol Sequence
        # Sobol sequences provide good space-filling properties
        n_initial = 64
        n_initial = min(n_initial, max(16, budget // 8))
        
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Sobol requires power of 2 for best performance, but handles others
        # We generate n_initial points
        sobol_samples = sampler.random(n_initial)
        # Map from [0, 1]^dim to [-5, 5]^dim
        initial_points = sobol_samples * 10.0 - 5.0
        
        initial_values = []
        for pt in initial_points:
            if eval_count >= budget or time.time() > deadline:
                break
            val = counting_func(pt)
            initial_values.append((val, pt))
            
        if not initial_values:
            # Fallback if something went wrong
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0
            return best_x, best_f

        # Sort by value to find best candidates
        initial_values.sort(key=lambda x: x[0])
        
        # Phase 2: Multi-Start CMA-ES
        # Select top distinct candidates
        # We want to avoid starting CMA-ES from points that are very close to each other
        candidates = []
        min_distance = 0.5 # Threshold for considering points "distinct"
        
        for val, pt in initial_values:
            is_distinct = True
            for c in candidates:
                if np.linalg.norm(pt - c) < min_distance:
                    is_distinct = False
                    break
            if is_distinct:
                candidates.append(pt)
            if len(candidates) >= 5:
                break

        # If we didn't get enough distinct candidates, just take the top ones
        if len(candidates) < 3:
             candidates = [pt for _, pt in initial_values[:3]]

        # Allocate remaining budget to CMA-ES runs
        remaining = budget - eval_count
        if remaining > 50 and candidates:
            
            # We will run CMA-ES from each candidate
            # Budget per run: roughly remaining / len(candidates), but capped
            # CMA-ES default popsize is 4 + 3 * log(n) ~ 13 for n=5
            # Let's use a fixed popsize of 15 for consistency
            popsize = 15
            
            # We want to ensure each run gets enough iterations
            # But also allow early stopping if time/budget is tight
            for i, x_start in enumerate(candidates):
                if eval_count >= budget or time.time() > deadline:
                    break
                
                # Calculate budget for this specific run
                # Leave some buffer for final polish
                runs_left = len(candidates) - i
                alloc_per_run = (remaining - runs_left * 10) // runs_left 
                if alloc_per_run < 30:
                    alloc_per_run = 30 # Minimum viable run
                
                # Ensure we don't exceed total remaining
                local_budget = min(alloc_per_run, budget - eval_count)
                
                if local_budget < 20:
                    break

                try:
                    # CMA-ES options
                    # sigma0: initial step size. 0.5 is a reasonable start for [-5,5]
                    # bounds: enforce box constraints
                    res = scipy.optimize.minimize(
                        counting_func,
                        x_start,
                        method='CMA',
                        options={
                            'maxiter': local_budget // popsize + 5, # Ensure enough generations
                            'maxfev': local_budget,
                            'popsize': popsize,
                            'sigma0': 0.5,
                            'bounds': [(-5.0, 5.0)] * dim,
                            'verbose': 0,
                            'tol': 1e-10
                        }
                    )
                except Exception:
                    continue

        # Phase 3: Final Polish
        # Use Nelder-Mead or CMA-ES with very tight tolerance on the global best
        remaining = budget - eval_count
        if remaining > 20 and best_x is not None:
            try:
                # Nelder-Mead is fast and good for final convergence
                res_nm = scipy.optimize.minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                # Update best if NM found something better (it should, as it uses counting_func)
            except Exception:
                pass

        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0

        return best_x, best_f
