import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

class Optimizer:
    """
    Optimizer using Differential Evolution followed by Nelder-Mead polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        calls = 0
        best_x = None
        best_f = float('inf')
        
        start_time = time.time()
        deadline = 290.0  # Safety margin from 300s limit

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            
            if time.time() > deadline:
                return float('inf')
            
            if calls >= self.budget:
                return float('inf')
            
            calls += 1
            try:
                val = func(x)
            except Exception:
                val = float('inf')
            
            if val < best_f:
                best_f = val
                best_x = np.array(x).copy()
                
            return val

        bounds = [(-5.0, 5.0)] * self.dim
        
        # Phase 1: Differential Evolution
        # popsize=10 -> 50 individuals
        # maxiter=8 -> 400 calls approx
        try:
            res_de = differential_evolution(
                wrapped_func,
                bounds,
                maxiter=8,
                popsize=10,
                tol=1e-6,
                seed=42
            )
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = res_de.x
        except Exception:
            pass

        # Phase 2: Local Polish with Nelder-Mead
        if calls < self.budget and best_x is not None:
            try:
                res_nm = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x
            except Exception:
                pass

        return best_x, best_f
