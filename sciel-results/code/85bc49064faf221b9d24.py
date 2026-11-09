import numpy as np
import time
import scipy.optimize

class Optimizer:
    """
    (1+1)-ES with adaptive sigma + Nelder-Mead polish.
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

        # Split budget: 30% ES, 70% NM
        es_budget = int(budget * 0.3)
        nm_budget = budget - es_budget
        if es_budget < 10:
            es_budget = max(1, budget // 2)
            nm_budget = budget - es_budget

        rng = np.random.default_rng(123)

        # Phase 1: (1+1)-ES with faster adaptation
        x = rng.uniform(-4.0, 4.0, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 2.0
        calls_used = 1
        success_count = 0
        total_count = 0

        for i in range(es_budget - 1):
            if time.time() > deadline or calls_used >= budget:
                break
            z = rng.standard_normal(dim)
            y = np.clip(x + sigma * z, lower, upper)
            fy = func(y)
            calls_used += 1
            total_count += 1
            if fy < fx:
                x, fx = y, fy
                success_count += 1
                sigma = min(sigma * 1.25, 5.0)
            else:
                sigma = max(sigma * 0.75, 0.01)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            
            # Adaptive sigma adjustment based on success rate
            if total_count % 20 == 0 and total_count > 0:
                success_rate = success_count / total_count
                if success_rate > 0.6:
                    sigma = min(sigma * 1.1, 5.0)
                elif success_rate < 0.3:
                    sigma = max(sigma * 0.9, 0.01)

        # Phase 2: Nelder-Mead with tighter tolerances
        x_start = best_x.copy()
        nm_best_x = x_start.copy()
        nm_best_f = best_fx

        def wrapped(x):
            nonlocal nm_best_x, nm_best_f, calls_used
            if calls_used >= budget or time.time() > deadline:
                return 1e6
            x = np.clip(x, lower, upper)
            try:
                fv = func(x)
            except Exception:
                fv = 1e6
            calls_used += 1
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
                p[i] += 0.05
                simplex.append(np.clip(p, lower, upper))
            scipy.optimize.minimize(
                wrapped, x_start, method='Nelder-Mead',
                options={'maxiter': nm_budget * 10, 'xatol': 1e-10,
                         'fatol': 1e-10, 'adaptive': True,
                         'initial_simplex': simplex},
                callback=cb)
        except Exception:
            pass

        return nm_best_x, nm_best_f
