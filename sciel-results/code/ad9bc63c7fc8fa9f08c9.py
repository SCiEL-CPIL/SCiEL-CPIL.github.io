import numpy as np
import time
import scipy.optimize

class Optimizer:
    """
    Multi-Start (1+1)-ES followed by Nelder-Mead polish.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        lower, upper = self.bounds

        # Split budget: 40% Multi-Start ES, 60% NM
        es_budget = int(budget * 0.4)
        nm_budget = budget - es_budget
        
        # Ensure minimums
        if es_budget < 10:
            es_budget = max(1, budget // 2)
            nm_budget = budget - es_budget
            
        rng = np.random.default_rng(123)
        
        # Phase 1: Multi-Start (1+1)-ES
        # We run multiple independent (1+1)-ES trajectories in parallel (conceptually)
        # but sequentially to respect the budget.
        # Number of starts: 5
        n_starts = 5
        calls_per_start = es_budget // n_starts
        if calls_per_start < 5:
            n_starts = max(1, es_budget // 5)
            calls_per_start = es_budget // n_starts

        best_fx_global = np.inf
        best_x_global = None
        calls_used = 0

        for s in range(n_starts):
            if time.time() > deadline or calls_used >= budget:
                break
                
            # Initialize start point
            x = rng.uniform(-4.0, 4.0, dim)
            fx = func(x)
            calls_used += 1
            
            if np.isnan(fx) or np.isinf(fx):
                continue
                
            if fx < best_fx_global:
                best_fx_global = fx
                best_x_global = x.copy()
                
            sigma = 2.0
            
            # Run (1+1)-ES for this start
            for i in range(calls_per_start - 1):
                if time.time() > deadline or calls_used >= budget:
                    break
                    
                z = rng.standard_normal(dim)
                y = np.clip(x + sigma * z, lower, upper)
                fy = func(y)
                calls_used += 1
                
                if np.isnan(fy) or np.isinf(fy):
                    continue
                    
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.1, 5.0)
                else:
                    sigma = max(sigma * 0.9, 0.05)
                    
                if fx < best_fx_global:
                    best_fx_global = fx
                    best_x_global = x.copy()

        # If no valid point found, fallback to center
        if best_x_global is None:
            best_x_global = np.zeros(dim)
            best_fx_global = func(best_x_global)
            calls_used += 1

        # Phase 2: Nelder-Mead
        x_start = best_x_global.copy()
        nm_calls = 0
        nm_best_x = x_start.copy()
        nm_best_f = best_fx_global

        def wrapped(x):
            nonlocal nm_calls, nm_best_x, nm_best_f, calls_used
            if calls_used >= budget or time.time() > deadline:
                return 1e6
            x = np.clip(x, lower, upper)
            try:
                fv = func(x)
            except Exception:
                fv = 1e6
            calls_used += 1
            nm_calls += 1
            if np.isnan(fv) or np.isinf(fv):
                return 1e6
            if fv < nm_best_f:
                nm_best_f = fv
                nm_best_x = x.copy()
            return fv

        def cb(xk):
            return calls_used >= budget or time.time() > deadline

        try:
            simplex = [x_start.copy()]
            for i in range(dim):
                p = x_start.copy()
                p[i] += 0.1
                simplex.append(np.clip(p, lower, upper))
            scipy.optimize.minimize(
                wrapped, x_start, method='Nelder-Mead',
                options={'maxiter': nm_budget * 5, 'xatol': 1e-12,
                         'fatol': 1e-12, 'adaptive': True,
                         'initial_simplex': simplex},
                callback=cb)
        except Exception:
            pass

        return nm_best_x, nm_best_f
