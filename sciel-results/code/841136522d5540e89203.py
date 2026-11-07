import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        # Wrapper to track calls and enforce deadline
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = func(x)
            except Exception:
                val = best_f
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float)
            return val

        # Phase 1: Multi-start L-BFGS-B
        # Use a small number of random starting points to identify promising basins
        # f22 (Gallagher 21 peaks) has multiple local minima, so multi-start is important.
        
        rng = np.random.default_rng(42)
        
        # Try a few random starts
        n_starts = 5
        start_points = rng.uniform(-5, 5, (n_starts, dim))
        
        # Also try the origin and a few structured points
        start_points = np.vstack([
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            start_points
        ])
        
        # We have 500 budget. Let's use ~300 for multi-start L-BFGS-B and 200 for polish.
        # Each L-BFGS-B run might use 50-100 evaluations.
        # Let's allocate budget per start.
        
        for i in range(len(start_points)):
            if calls >= budget - 100 or time.time() > deadline - 10:
                break
            
            x0 = start_points[i]
            # L-BFGS-B with bounds
            # maxfun is the maximum number of function evaluations
            # We want to limit it per start.
            # Let's give each start up to 50 evaluations.
            maxfun = 50
            if calls + maxfun > budget - 100:
                maxfun = budget - 100 - calls
            
            if maxfun < 10:
                break
                
            res = minimize(
                wrapped_func, 
                x0, 
                method='L-BFGS-B', 
                bounds=[(-5, 5)] * dim,
                options={'maxfun': maxfun, 'maxiter': 100}
            )
            
            # Update best if L-BFGS-B found something better
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        
        # Phase 2: Polish the best point found so far
        # Use Nelder-Mead or a simple (1+1)-ES for fine-tuning
        remaining_budget = budget - calls
        if remaining_budget > 20 and time.time() < deadline - 5:
            # Use Nelder-Mead for local polish
            # maxiter for Nelder-Mead is number of iterations, not evals.
            # But we can use the wrapper to limit evals.
            # Let's just use a simple (1+1)-ES or Nelder-Mead with a tight budget.
            
            # Nelder-Mead can be sensitive. Let's try it.
            # Or just use L-BFGS-B again from the best point with a smaller budget?
            # Or a simple coordinate descent?
            
            # Let's use Nelder-Mead with a custom wrapper that stops after remaining_budget/2 evals.
            # Actually, the wrapper already tracks global calls.
            
            res = minimize(
                wrapped_func,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': remaining_budget // 2, 'xatol': 1e-8, 'fatol': 1e-12}
            )
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        
        # Phase 3: If we still have budget, do a few more random restarts or a DE-like search?
        # Given the time constraint and the nature of f22, multi-start L-BFGS-B is usually strong.
        # Let's add a small DE phase if we have time and budget.
        
        if calls < budget - 50 and time.time() < deadline - 20:
            # Simple DE for exploration
            np_size = 10
            pop = rng.uniform(-5, 5, (np_size, dim))
            # Ensure best_x is in population
            pop[0] = best_x
            
            f_pop = np.array([wrapped_func(x) for x in pop])
            idx = np.argmin(f_pop)
            best_x = pop[idx].copy()
            best_f = f_pop[idx]
            
            F = 0.8
            CR = 0.9
            max_gen = (budget - calls - 10) // np_size
            
            for g in range(max_gen):
                if calls >= budget - 10 or time.time() > deadline - 5:
                    break
                for i in range(np_size):
                    if calls >= budget - 10 or time.time() > deadline - 5:
                        break
                    
                    # Select two distinct random indices
                    others = [j for j in range(np_size) if j != i]
                    r1, r2 = rng.choice(others, size=2, replace=False)
                    
                    v = pop[r1] + F * (pop[r2] - pop[i])
                    
                    trial = pop[i].copy()
                    cross_mask = rng.random(dim) < CR
                    if not np.any(cross_mask):
                        j = rng.integers(0, dim)
                        cross_mask[j] = True
                    trial[cross_mask] = v[cross_mask]
                    trial = np.clip(trial, -5, 5)
                    
                    f_trial = wrapped_func(trial)
                    
                    if f_trial < f_pop[i]:
                        pop[i] = trial
                        f_pop[i] = f_trial
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = trial.copy()
        
        return best_x, best_f
