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
        deadline = start_time + 280

        rng = np.random.default_rng(42)
        call_count = [0]
        best_x = None
        best_f = np.inf

        def safe_call(x):
            if call_count[0] >= budget:
                return np.inf
            call_count[0] += 1
            return func(x)

        # Phase 1: Multi-start (1+1)-ES
        # Allocate 90% of budget to ES
        es_budget = int(budget * 0.9)
        n_starts = 10
        steps_per_start = es_budget // n_starts
        if steps_per_start < 10:
            steps_per_start = 10
            n_starts = es_budget // steps_per_start

        for i in range(n_starts):
            if time.time() > deadline or call_count[0] >= es_budget:
                break

            # Diverse starting points
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-5, 5, dim)
            
            # Clip to box
            x = np.clip(x, -5, 5)
            
            fx = safe_call(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 1.0
            for _ in range(steps_per_start):
                if time.time() > deadline or call_count[0] >= es_budget:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_call(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                
                if sigma < 1e-8:
                    break

        # Phase 2: L-BFGS-B Polish
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            remaining = budget - call_count[0]
            if remaining > 5:
                try:
                    res = minimize(
                        safe_call,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': remaining, 'maxls': 10}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
