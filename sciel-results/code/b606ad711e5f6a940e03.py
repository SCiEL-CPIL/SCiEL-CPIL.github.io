import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Weight vectors for 3 objectives
        # Simplex vertices and edge midpoints
        weights = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
            np.array([1/3, 1/3, 1/3])
        ]
        
        n_weights = len(weights)
        budget_per_weight = self.budget // n_weights
        
        results = []
        evals_used = 0
        
        for w in weights:
            if evals_used >= self.budget:
                break
            
            # Generate random start
            x0 = lo + (hi - lo) * np.random.random(self.dim)
            
            # Define scalarized objective
            def obj(x):
                nonlocal evals_used
                f = problem(x)
                evals_used += 1
                return np.dot(w, f)
            
            # Run Nelder-Mead
            # options: maxfev limits the number of function calls
            try:
                res = minimize(obj, x0, method='Nelder-Mead', options={'maxfev': budget_per_weight, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.success or res.fun < np.inf:
                    results.append(res.x)
            except Exception:
                pass
        
        return results
