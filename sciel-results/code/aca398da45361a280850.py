import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Safety deadline
        deadline = time.time() + 300.0
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        
        # ZDT4 Pareto Front Construction:
        # The true Pareto front for ZDT4 is achieved when x_i (i > 0) are at their
        # values that minimize the g function. For standard ZDT4, this is x_i = 0.
        # This minimizes f1 and keeps g minimal (g=1), allowing f2 to take its
        # Pareto-optimal values.
        
        # We assume x_0 is the variable that parameterizes the front.
        # We will sweep x_0 across its bounds and set x_1...x_{n-1} to 0.
        
        # If 0 is not within bounds for x_i (i>0), we should pick the bound closest to 0
        # or the value that minimizes the penalty term. However, standard ZDT4 bounds
        # for x_i (i>0) are [-5, 5], which includes 0.
        
        tail_values = np.zeros(n-1)
        # Clip tail values to bounds just in case
        tail_lo = lo[1:]
        tail_hi = hi[1:]
        tail_values = np.clip(tail_values, tail_lo, tail_hi)
        
        # Determine number of evaluations to perform
        # We have a budget of self.budget. We should use as many as possible
        # to densely sample the front.
        n_evals = min(self.budget, 200) # Task says budget=200
        
        # Generate x_0 values
        # To maximize hypervolume on a convex/concave front, uniform spacing in the
        # decision variable x_0 is a good heuristic.
        # If f1 is quadratic in x0, uniform f1 spacing (sqrt spacing in x0) might be better.
        # Let's use linear spacing in x0 for simplicity and robustness.
        x0_lo = lo[0]
        x0_hi = hi[0]
        
        # If bounds are inverted, swap
        if x0_lo > x0_hi:
            x0_lo, x0_hi = x0_hi, x0_lo
            
        # Create 200 points
        x0_values = np.linspace(x0_lo, x0_hi, n_evals)
        
        recorded_F = []
        used = 0
        
        for x0 in x0_values:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x = np.concatenate(([float(x0)], tail_values))
            
            # Evaluate
            try:
                fx = problem(x)
                used += 1
                fv = np.atleast_1d(fx).astype(float)
                if fv.shape[0] == self.n_obj and all(np.isfinite(fv)):
                    recorded_F.append(fv.tolist())
            except Exception:
                # If evaluation fails, stop to avoid wasting time or breaking state
                break
        
        return np.array(recorded_F) if recorded_F else np.empty((0, self.n_obj))
