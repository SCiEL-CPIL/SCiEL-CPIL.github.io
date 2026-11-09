import numpy as np
from scipy.optimize import basinhopping

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        
        bounds = [(-5.0, 5.0)] * self.dim
        x0 = rng.uniform(-5.0, 5.0, self.dim)
        
        minimizer_kwargs = {
            'method': 'Nelder-Mead',
            'options': {
                'maxiter': 45,
                'xatol': 1e-4,
                'fatol': 1e-4
            }
        }
        
        res = basinhopping(
            func,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=8,
            stepsize=2.0,
            seed=0
        )
        
        return res.x, res.fun
