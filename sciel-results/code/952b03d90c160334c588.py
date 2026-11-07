import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        budget = self.budget

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

        def run_lbfgsb(x0, max_iter, gtol=1e-10):
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
                        'gtol': gtol
                    }
                )
            except Exception:
                pass

        # Phase 1: Dense Sobol quasi-random exploration to map the landscape
        # Use Sobol sequence for deterministic, space-filling coverage
        sobol_sample_count = 64  # 2^6 points, good coverage for 5D
        sobol = qmc.Sobol(d=n, scramble=True, seed=42)
        sobol_points = sobol.random(sobol_sample_count)
        
        # Map from [0,1]^n to [-5,5]^n
        sobol_points = sobol_points * 10.0 - 5.0
        
        # Evaluate all Sobol points to find promising regions
        for point in sobol_points:
            if calls >= budget or time.time() >= deadline:
                break
            evaluate(point)
        
        # Phase 2: Multi-restart gradient-based refinement from top Sobol points
        # Collect evaluated points sorted by function value
        evaluated_points = []
        for point in sobol_points[:min(calls, len(sobol_points))]:
            if calls >= budget or time.time() >= deadline:
                break
            fx = evaluate(point)
            evaluated_points.append((fx, point))
        
        # Sort by function value and take top K for refinement
        evaluated_points.sort(key=lambda x: x[0])
        top_k = min(8, len(evaluated_points))
        
        # Refine from the best Sobol points using L-BFGS-B and NM
        for fx, x0 in evaluated_points[:top_k]:
            if calls >= budget or time.time() >= deadline:
                break
            
            # L-BFGS-B for smooth convergence
            run_lbfgsb(x0, max_iter=30)
            
            # Nelder-Mead for robustness
            if calls < budget and time.time() < deadline:
                run_nm(x0, max_iter=20)
        
        # Phase 3: Additional refinement from best found point
        if best_x is not None and calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                run_lbfgsb(best_x, max_iter=remaining, gtol=1e-12)
            
            if best_x is not None and calls < budget and time.time() < deadline:
                remaining = budget - calls
                if remaining > 5:
                    run_nm(best_x, max_iter=remaining, xatol=1e-10, fatol=1e-12)
        
        # Phase 4: Final polish with tighter tolerances
        if best_x is not None and calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining > 3:
                run_lbfgsb(best_x, max_iter=remaining, gtol=1e-14)
            
            if best_x is not None and calls < budget and time.time() < deadline:
                remaining = budget - calls
                if remaining > 2:
                    run_nm(best_x, max_iter=remaining, xatol=1e-12, fatol=1e-14)

        return best_x, best_f
