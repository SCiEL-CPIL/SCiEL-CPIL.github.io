import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Initial point
        x0 = rng.uniform(-5, 5, dim)
        
        # Use CMA-ES via SciPy
        # maxfev is the maximum number of function evaluations
        res = minimize(func, x0, method='CMA', options={'maxfev': budget, 'tol': 1e-12})
        
        return res.x, res.fun
