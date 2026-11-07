import numpy as np
import time
import scipy.optimize as opt


class Optimizer:
    """Multi-Start Nelder-Mead optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_fx = float('inf')
        total_evals = 0

        def evaluate(x):
            nonlocal total_evals, best_x, best_fx
            if total_evals >= budget:
                return 1e10
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            total_evals += 1
            if f < best_fx:
                best_fx = f
                best_x = x.copy()
            return f

        def run_nm(x0, max_iter):
            nonlocal total_evals, best_x, best_fx
            if total_evals >= budget or time.time() > deadline:
                return

            try:
                res = opt.minimize(
                    lambda x: evaluate(x),
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 1: Global Search with Multiple Restarts
        num_restarts = 5
        # Estimate max_iter for Nelder-Mead
        # In 5D, 1 iteration ~ 6-7 function calls.
        # We want to use ~80% of budget for exploration.
        # 400 evals / 5 restarts = 80 evals per restart.
        # 80 evals / 7 evals/iter ~ 11 iterations.
        # Let's allow 20 iterations to be safe, but budget guard will stop it.
        max_iter_exploration = 20

        for i in range(num_restarts):
            if total_evals >= budget or time.time() > deadline:
                break
            
            # Diverse starting points
            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = rng.uniform(-3, 3, dim)
            elif i == 2:
                x0 = rng.uniform(-5, 5, dim)
            elif i == 3:
                x0 = rng.uniform(-1, 1, dim)
            else:
                x0 = rng.uniform(-4, 4, dim)
            
            # Initial evaluation of x0
            f0 = evaluate(x0)
            
            run_nm(x0, max_iter_exploration)

        # Phase 2: Local Polish
        if best_x is not None and total_evals < budget and time.time() < deadline:
            remaining_budget = budget - total_evals
            # Estimate max_iter for polish
            # 1 iteration ~ 6 evals.
            max_iter_polish = remaining_budget // 5
            
            if max_iter_polish > 0:
                run_nm(best_x, max_iter_polish)

        return best_x, best_fx
