import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_nm(x0, max_iter, xatol=1e-8, fatol=1e-10):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                
                minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': xatol,
                        'fatol': fatol,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        def run_lbfgsb(x0, max_iter=200, gtol=1e-10):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                
                minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': max_iter,
                        'gtol': gtol,
                        'maxls': 50
                    }
                )
            except Exception:
                pass

        # Phase 1: Diverse Exploration with Nelder-Mead
        num_exploration = 10
        calls_per_exploration = max(15, (budget - 150) // num_exploration)
        
        starts = [np.zeros(n)]
        for _ in range(num_exploration - 1):
            starts.append(rng.uniform(-4.5, 4.5, size=n))
            
        for x0 in starts:
            if calls >= budget or time.time() >= deadline:
                break
            run_nm(x0, calls_per_exploration)

        # Phase 2: Polish with L-BFGS-B from best point
        if best_x is not None and calls < budget and time.time() < deadline:
            remaining = budget - calls
            # Use L-BFGS-B for efficient gradient-based convergence
            run_lbfgsb(best_x.copy(), max_iter=min(remaining, 150), gtol=1e-12)
        
        # Phase 3: Final high-precision polish
        if best_x is not None and calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining > 20:
                # Try L-BFGS-B with very tight tolerance
                run_lbfgsb(best_x.copy(), max_iter=min(remaining, 100), gtol=1e-14)
            
            # Final NM polish with tight tolerances
            if calls < budget and time.time() < deadline:
                remaining = budget - calls
                if remaining > 10:
                    run_nm(best_x.copy(), max_iter=remaining, xatol=1e-12, fatol=1e-14)

        return best_x, best_f
