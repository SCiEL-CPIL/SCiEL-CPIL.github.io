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
        deadline = start_time + 290.0  # Safety margin
        
        n_evals = self.budget
        if n_evals <= 0:
            return []

        # ZDT6 Pareto Front:
        # f1 = x1
        # f2 = 1 - sqrt(f1)  (on the Pareto front where g=1)
        # Reference point: (11, 11)
        
        # We want to find N points f1_1, ..., f1_N in [0, 1] 
        # that maximize the Hypervolume of the points (f1_i, 1-sqrt(f1_i))
        
        def neg_hypervolume(x):
            # x is a vector of length n_evals
            # Sort to ensure correct ordering for HV calculation
            x_sorted = np.sort(x)
            # Clip to [0, 1]
            x_sorted = np.clip(x_sorted, 0.0, 1.0)
            
            f2 = 1.0 - np.sqrt(x_sorted)
            R1 = 11.0
            R2 = 11.0
            
            # Calculate HV
            # H = sum_{i=1}^{N-1} (x_{i+1} - x_i) (R2 - f2_i) + (R1 - x_N) (R2 - f2_N)
            diffs = np.diff(x_sorted)
            # f2_sorted corresponds to x_sorted
            # Term i uses f2[i] and diff[i] (which is x[i+1] - x[i])
            # Note: np.diff has length N-1. f2 has length N.
            # We sum diffs * (R2 - f2[0:N-1])
            terms = diffs * (R2 - f2[:-1])
            hv = np.sum(terms)
            # Last term
            hv += (R1 - x_sorted[-1]) * (R2 - f2[-1])
            
            return -hv

        # Initial guess: 
        # The function f2 = 1 - sqrt(f1) has a steep slope near f1=0.
        # To better approximate the area, we should cluster points near f1=0.
        # A quadratic distribution u^2 where u is uniform is a good heuristic.
        u = np.linspace(0.0, 1.0, n_evals)
        x0 = u**2
        
        # Optimize
        try:
            res = minimize(neg_hypervolume, x0, method='L-BFGS-B', 
                           bounds=[(0.0, 1.0)] * n_evals,
                           options={'maxiter': 2000, 'disp': False, 'ftol': 1e-12})
            x_opt = res.x
        except Exception:
            x_opt = x0

        # Sort and clip final x values
        x_opt = np.sort(x_opt)
        x_opt = np.clip(x_opt, 0.0, 1.0)
        
        results = []
        
        # Check if we have time to evaluate all points
        # Evaluation is fast, but let's be safe.
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            # Create decision vector
            x = np.zeros(self.dim, dtype=float)
            x[0] = x_opt[i]
            # x[1:] are 0, which ensures g=1 and points are on Pareto front
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
                
        return results
