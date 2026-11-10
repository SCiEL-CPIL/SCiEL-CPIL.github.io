import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        self.eval_count = 0
        start_time = time.time()
        deadline = 325.0 # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        def evaluate(x):
            if self.eval_count >= self.budget:
                return np.array([np.inf]*self.n_obj)
            self.eval_count += 1
            return problem(x)

        # 1. Random Initialization
        n_init = 20
        X = lo + range_ * np.random.random((n_init, self.dim))
        F = np.array([evaluate(x) for x in X])
        
        # 2. Find Non-Dominated Points
        def is_non_dominated(f, F):
            for j in range(len(F)):
                if j == np.where(f == F[j])[0][0]: continue
                if np.all(F[j] <= f) and np.any(F[j] < f):
                    return False
            return True

        nd_indices = [i for i in range(len(F)) if is_non_dominated(F[i], F)]
        
        # If no ND points found (shouldn't happen with random), take all
        if not nd_indices:
            nd_indices = list(range(len(F)))
            
        # Limit number of points to polish to avoid wasting budget
        max_polish = 5
        if len(nd_indices) > max_polish:
            # Sort by sum of objectives to pick "best" ones
            nd_indices.sort(key=lambda i: np.sum(F[i]))
            nd_indices = nd_indices[:max_polish]

        # 3. Local Polish using Nelder-Mead on Scalarized Objectives
        # We will use Weighted Sum with random weights for diversity
        
        remaining_budget = self.budget - self.eval_count
        evals_per_polish = max(10, remaining_budget // max_polish)
        
        for idx in nd_indices:
            if self.eval_count >= self.budget:
                break
            if time.time() > deadline - 5:
                break
                
            x0 = X[idx]
            # Random weights
            w = np.random.random(self.n_obj)
            w /= np.sum(w)
            
            def obj_scalar(x):
                if self.eval_count >= self.budget:
                    return np.inf
                f = evaluate(x)
                return np.dot(f, w)
                
            try:
                res = minimize(obj_scalar, x0, method='Nelder-Mead', 
                               options={'maxiter': evals_per_polish // (self.dim + 1), 
                                        'xatol': 1e-4, 'fatol': 1e-4})
            except Exception:
                pass

        # 4. Use remaining budget for more random sampling or polishing
        if self.eval_count < self.budget - 10:
             # Do a few more random restarts with short polish
             n_extra = (self.budget - self.eval_count) // 20
             for _ in range(n_extra):
                 if self.eval_count >= self.budget:
                     break
                 if time.time() > deadline - 2:
                     break
                 x0 = lo + range_ * np.random.random(self.dim)
                 w = np.random.random(self.n_obj)
                 w /= np.sum(w)
                 
                 def obj_scalar2(x):
                     if self.eval_count >= self.budget:
                         return np.inf
                     f = evaluate(x)
                     return np.dot(f, w)
                     
                 try:
                     res = minimize(obj_scalar2, x0, method='Nelder-Mead', 
                                    options={'maxiter': 5, 'xatol': 1e-3, 'fatol': 1e-3})
                 except Exception:
                     pass

        return None
