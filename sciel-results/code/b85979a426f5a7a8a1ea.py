import numpy as np
from scipy.optimize import dual_annealing, minimize
from scipy.stats.qmc import LatinHypercube
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290
        
        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return np.inf
            return func(x)

        bounds = [(-5.0, 5.0)] * dim
        
        # Strategy: Use Dual Annealing which combines global and local search.
        # It is robust for smooth functions like Rosenbrock.
        try:
            res = dual_annealing(
                wrapped_func,
                bounds,
                maxiter=100, 
                maxfev=budget, 
                restarts=10,   
                seed=42,
                initial_temp=5230.0,
                visit=2.62,
                boltzmann=11.41,
                step_decay=0.98,
                coolinc=0.037,
                update_period=50,
                forcefire=True,
                disp=False
            )
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
                
        except Exception:
            # Fallback to LHS + L-BFGS-B if dual_annealing fails or times out
            if time.time() > deadline:
                if best_x is None:
                    best_x = np.zeros(dim)
                    best_f = wrapped_func(best_x)
                return best_x, best_f
            
            n_lhs = 50
            sampler = LatinHypercube(d=dim, seed=42)
            lhs_points = sampler.random(n=n_lhs)
            lhs_points = lhs_points * 8.0 - 4.0
            
            lhs_results = []
            for x in lhs_points:
                if call_count[0] >= budget - 100:
                    break
                f_val = wrapped_func(x)
                lhs_results.append((f_val, x))
                
            if lhs_results:
                lhs_results.sort(key=lambda item: item[0])
                best_f, best_x = lhs_results[0]
                
                starts = [p[1] for p in lhs_results[:3]]
                for x0 in starts:
                    if time.time() > deadline or call_count[0] >= budget - 20:
                        break
                    try:
                        res2 = minimize(wrapped_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50, 'disp': False})
                        if res2.fun < best_f:
                            best_f = res2.fun
                            best_x = res2.x
                    except Exception:
                        pass
            else:
                best_x = np.zeros(dim)
                best_f = wrapped_func(best_x)

        return best_x, best_f
