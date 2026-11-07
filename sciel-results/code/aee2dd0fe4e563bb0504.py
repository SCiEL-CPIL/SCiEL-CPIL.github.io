import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s, we aim to finish by 315s
        deadline = start_time + 315
        
        rng = np.random.default_rng(42)
        
        # 1. Initial diverse sampling
        # Rosenbrock is narrow. Random uniform is often poor.
        # We do a small batch of random starts, pick the best, then polish.
        n_initial = 20
        X = np.vstack([
            np.zeros(self.dim),
            np.ones(self.dim) * 0.5,
            np.ones(self.dim) * -0.5,
            rng.uniform(-5, 5, (n_initial - 3, self.dim))
        ])
        F = np.array([func(x) for x in X])
        
        best_idx = np.argmin(F)
        best_x = X[best_idx].copy()
        best_f = F[best_idx]
        
        calls_used = n_initial
        remaining = self.budget - calls_used
        
        if time.time() > deadline - 5:
            return best_x, best_f

        # 2. Local Refinement using Powell's Method
        # Powell is derivative-free and uses line searches along directions.
        # This is robust for Rosenbrock as it doesn't rely on gradient estimates
        # which can be noisy or ill-conditioned in the narrow valley.
        # We wrap the function to count calls and enforce time/budget limits.
        
        call_count = [0]
        
        def wrapper(x):
            if call_count[0] >= remaining:
                raise ValueError("Budget exceeded")
            if time.time() > deadline - 5:
                raise ValueError("Time exceeded")
            call_count[0] += 1
            return func(x)
        
        try:
            # Powell method with default options, but we limit iterations
            # to ensure we don't hang. 
            # maxiter is the maximum number of iterations (directions updated)
            # maxfun is the maximum number of function evaluations
            res = minimize(wrapper, best_x, method='Powell', 
                           options={'maxiter': 100, 'maxfun': remaining, 'xtol': 1e-8, 'ftol': 1e-8})
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                
        except Exception:
            pass
        
        remaining = self.budget - (calls_used + call_count[0])
        
        if time.time() > deadline - 10:
            return best_x, best_f
        
        # 3. If budget remains, perform additional local optimizations from perturbed starts
        # using Powell again, as it is robust and derivative-free.
        
        if remaining > 50:
            n_restarts = min(5, remaining // 50)
            for i in range(n_restarts):
                if time.time() > deadline - 10:
                    break
                if remaining < 50:
                    break
                
                # Perturb the best point with a larger step to escape local minima
                # if stuck, or explore nearby basins.
                perturbation = rng.normal(0, 1.0, self.dim)
                start_x = np.clip(best_x + perturbation, -5, 5)
                
                restart_call_count = [0]
                
                def restart_wrapper(x):
                    if restart_call_count[0] >= remaining:
                        raise ValueError("Budget exceeded")
                    if time.time() > deadline - 5:
                        raise ValueError("Time exceeded")
                    restart_call_count[0] += 1
                    return func(x)
                
                try:
                    res_r = minimize(restart_wrapper, start_x, method='Powell',
                                     options={'maxiter': 50, 'maxfun': remaining, 'xtol': 1e-8, 'ftol': 1e-8})
                    
                    if res_r.fun < best_f:
                        best_f = res_r.fun
                        best_x = res_r.x.copy()
                        
                except Exception:
                    pass
                
                remaining = self.budget - (calls_used + call_count[0] + restart_call_count[0])
        
        return best_x, best_f
