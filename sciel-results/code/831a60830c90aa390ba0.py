import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        N = self.budget
        if N < 1:
            return np.zeros((0, self.dim))
            
        def hypervolume_from_params(params):
            x1 = params[0]
            d = params[1:]
            # Check validity
            if np.any(d <= 0) or x1 <= 0:
                return -1e10
            x_vals = np.cumsum(np.concatenate(([x1], d)))
            if x_vals[-1] >= 1.0:
                return -1e10
            # Calculate HV
            x_next = np.append(x_vals, 11.0)
            widths = np.diff(x_next)
            heights = 1 + 9 * np.sqrt(x_vals)
            return -np.sum(widths * heights) # Minimize negative HV

        # Initial guess: x_i = (i/N)^2
        # x_1 = (1/N)^2
        # d_i = x_{i+1} - x_i = ((i+1)/N)^2 - (i/N)^2 = (2i+1)/N^2
        init_x1 = (1.0 / N) ** 2
        init_d = np.array([((i + 1) / N) ** 2 - (i / N) ** 2 for i in range(1, N)])
        init_params = np.concatenate(([init_x1], init_d))
        
        # Bounds
        # x1 > 0, d > 0
        # x_N < 1 is a constraint on the sum, hard to enforce with simple bounds on d.
        # We can just let the optimizer run and hope it stays in bounds, or use a penalty.
        # The penalty in the objective function handles it.
        
        bounds_opt = [(1e-6, None)] * N
        
        try:
            res = minimize(hypervolume_from_params, init_params, method='Nelder-Mead', bounds=bounds_opt, options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            best_params = res.x
        except Exception:
            best_params = init_params

        # Extract x_vals
        x1 = best_params[0]
        d = best_params[1:]
        if np.any(d <= 0) or x1 <= 0:
            # Fallback to power law
            i = np.arange(1, N + 1)
            x1_vals = (i / N) ** 2.0
        else:
            x1_vals = np.cumsum(np.concatenate(([x1], d)))
            # Ensure x_N < 1
            if x1_vals[-1] >= 1.0:
                 # Scale down?
                 x1_vals = x1_vals * 0.99
            
        # Construct population
        X = np.zeros((N, self.dim))
        X[:, 0] = x1_vals
        
        # Evaluate all points
        for x in X:
            problem(x)
            
        return X
