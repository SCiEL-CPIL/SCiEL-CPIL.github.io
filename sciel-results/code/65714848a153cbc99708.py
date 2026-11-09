import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        eval_count = 0
        
        def eval_wrapper(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return np.zeros(n_obj)
            x = np.clip(x, lo, hi)
            val = np.array(problem(x))
            eval_count += 1
            return val

        # Strategy: Decomposition-based local search with LHS initialization
        n_runs = 5
        n_random = 10
        budget_for_runs = self.budget - n_random
        max_iter_per_run = max(10, budget_for_runs // n_runs)
        
        weights = []
        for i in range(n_runs):
            w = np.random.dirichlet(np.ones(n_obj))
            weights.append(w)
            
        x0_list = []
        for i in range(n_runs):
            x0 = lo + (hi - lo) * np.random.rand(dim)
            x0_list.append(x0)
            
        results = []
        
        for i in range(n_runs):
            if time.time() > deadline:
                break
            if eval_count >= self.budget - n_random:
                break
                
            w = weights[i]
            x0 = x0_list[i]
            
            def objective(x):
                f = eval_wrapper(x)
                return np.dot(w, f)
                
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter_per_run, 'xatol': 1e-3, 'fatol': 1e-3})
                if eval_count < self.budget - n_random:
                    f_best = eval_wrapper(res.x)
                    results.append((res.x, f_best))
            except Exception:
                if eval_count < self.budget - n_random:
                    f_x0 = eval_wrapper(x0)
                    results.append((x0, f_x0))

        for i in range(n_random):
            if time.time() > deadline:
                break
            if eval_count >= self.budget:
                break
            x_rand = lo + (hi - lo) * np.random.rand(dim)
            f_rand = eval_wrapper(x_rand)
            results.append((x_rand, f_rand))

        if results:
             return np.array([r[0] for r in results])
        return None
