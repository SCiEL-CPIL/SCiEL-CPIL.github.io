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



        # Phase 1: Diverse Exploration with Powell's method
        num_exploration = 12
        calls_per_exploration = max(8, (budget - 50) // num_exploration)
        
        # Create diverse starting points including origin and random samples
        starts = [np.zeros(n)]
        for _ in range(num_exploration - 1):
            starts.append(rng.uniform(-4.8, 4.8, size=n))
            
        def run_powell(x0, maxiter=200):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                
                minimize(
                    objective,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': maxiter,
                        'xtol': 1e-8,
                        'ftol': 1e-10
                    }
                )
            except Exception:
                pass
        
        for x0 in starts:
            if calls >= budget or time.time() >= deadline:
                break
            # Use Powell's method for derivative-free local search
            run_powell(x0, maxiter=calls_per_exploration)
            # Follow up with NM for fine-tuning if budget allows
            if calls < budget and time.time() < deadline:
                run_nm(x0, max_iter=max(5, calls_per_exploration // 3))

        # Phase 2: Polish with high-precision methods
        if best_x is not None and calls < budget and time.time() < deadline:
            remaining = budget - calls
            
            # Powell polish with tighter tolerance
            if remaining > 15:
                run_powell(best_x, maxiter=min(remaining, 100))
            
            # Final NM polish with very tight tolerance
            if best_x is not None and calls < budget and time.time() < deadline:
                remaining = budget - calls
                if remaining > 5:
                    run_nm(best_x, max_iter=remaining, xatol=1e-12, fatol=1e-14)

        return best_x, best_f
