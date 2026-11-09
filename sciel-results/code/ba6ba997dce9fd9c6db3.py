import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit. Use 280s to be safe.
        deadline = start_time + 280.0
        
        rng = np.random.default_rng(0)
        best_x = None
        best_f = np.inf
        evals_used = 0
        
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        
        # Phase 1: Exploration with Random Restarts
        # Use 60% of budget for exploration
        phase1_budget = int(self.budget * 0.6)
        num_restarts = 10
        # Distribute phase1_budget among restarts
        # Reserve some for the last restart or just cap it
        evals_per_restart = max(10, phase1_budget // num_restarts)
        
        for i in range(num_restarts):
            if evals_used >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x0 = rng.uniform(-5, 5, self.dim)
            
            # Estimate max iterations for Nelder-Mead
            # ~6 evals per iteration for dim=5
            max_iter = max(1, (evals_per_restart - 1) // 6)
            
            try:
                res = scipy.optimize.minimize(
                    func, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-3,
                        'fatol': 1e-3,
                        'adaptive': True
                    }
                )
            except Exception:
                continue
                
            evals_used += res.nfev
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        
        # Phase 2: Exploitation
        # Use remaining budget to refine the best solution found
        if best_x is not None and evals_used < self.budget:
            if time.time() > deadline:
                break
            
            remaining_budget = self.budget - evals_used
            # Nelder-Mead from a good start converges faster
            # Use most of remaining budget
            max_iter = max(1, (remaining_budget - 1) // 6)
            
            try:
                res = scipy.optimize.minimize(
                    func, 
                    best_x, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-5,
                        'fatol': 1e-5,
                        'adaptive': True
                    }
                )
            except Exception:
                pass # Keep previous best
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
            evals_used += res.nfev

        return best_x, best_f
