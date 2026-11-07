import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer using a Deterministic Global Optimization strategy with Sobol sequences.
    1. Generates the full evaluation sequence upfront using Sobol quasi-random points.
    2. Evaluates points in order, tracking the best found.
    3. If time permits, performs a final local polish on the best point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, use 290s to be safe
        deadline = start_time + 290
        
        n = self.dim
        total_budget = self.budget
        
        calls = 0
        best_x = np.zeros(n)
        best_f = np.inf
        
        # Helper to clip to box
        def clip(x):
            return np.clip(x, -5, 5)

        # Evaluate function wrapper
        def eval_func(x):
            nonlocal calls, best_f, best_x
            if calls >= total_budget or time.time() > deadline:
                return np.inf
            x = clip(x)
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Generate Sobol sequence for all evaluations
        # We reserve a small portion for local polish if time permits
        n_sobol = total_budget
        
        try:
            sampler = qmc.Sobol(d=n, seed=42)
            # Sobol requires power of 2 for best uniformity, but we can just take first N
            # Actually, for better distribution, we might want to use a power of 2 >= N and take first N
            # But standard practice is to just generate N points.
            # Let's use a slightly larger power of 2 to ensure good distribution properties if possible
            # 512 is the next power of 2 >= 500
            n_gen = 1
            while n_gen < n_sobol:
                n_gen *= 2
            
            samples = sampler.random(n_gen)
            # Take the first n_sobol points
            points = -5 + 10 * samples[:n_sobol]
        except Exception:
            # Fallback to random if Sobol fails
            points = np.random.uniform(-5, 5, (n_sobol, n))

        # Evaluate all Sobol points
        for p in points:
            if calls >= total_budget or time.time() > deadline:
                break
            eval_func(p)

        # If we have time and budget left, do a local polish
        if calls < total_budget and time.time() < deadline and best_f < np.inf:
            remaining = total_budget - calls
            # Only polish if we have significant time left
            time_left = deadline - time.time()
            if time_left > 10 and remaining > 10:
                # Use Nelder-Mead for polish
                def local_func(x):
                    if calls >= total_budget or time.time() > deadline:
                        raise ValueError("Budget Exceeded")
                    return eval_func(x)
                
                try:
                    maxiter = max(1, remaining // 3)
                    minimize(local_func, best_x, method='Nelder-Mead', 
                             options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
                except Exception:
                    pass

        return best_x, best_f
