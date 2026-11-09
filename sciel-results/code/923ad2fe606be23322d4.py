"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Global Halton Grid Search followed by Nelder-Mead Polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        # We split the budget: 
        # 80% for global exploration using a Halton sequence
        # 20% for local refinement using Nelder-Mead
        n_global = int(self.budget * 0.8)
        n_local = self.budget - n_global
        
        # 1. Global Exploration with Halton Sequence
        # Halton sequence is deterministic and low-discrepancy
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0
        
        # Generate Halton sequence points
        # We need a fast Halton generator or use a simple low-discrepancy approach
        # For dim=5, we can use the first 5 prime bases: 2, 3, 5, 7, 11
        
        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i = i // base
            return r

        bases = [2, 3, 5, 7, 11]
        
        for i in range(1, n_global + 1):
            if time.time() > deadline:
                break
            
            # Generate point in [0, 1]^dim
            point_unit = np.array([halton_sequence(i, base) for base in bases[:self.dim]])
            
            # Map to [-5, 5]^dim
            x = point_unit * 10.0 - 5.0
            
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            # If we have used too many calls, stop
            if calls_made >= self.budget:
                break

        # 2. Local Refinement with Nelder-Mead
        # Use the best point found as the initial guess
        # Nelder-Mead is good for polishing in the basin of attraction
        # We limit the number of function evaluations to n_local
        
        if calls_made < self.budget and time.time() < deadline:
            # Define a wrapper to count calls and enforce deadline
            local_calls = [0]
            local_limit = self.budget - calls_made
            
            def wrapped_func(x):
                local_calls[0] += 1
                if local_calls[0] > local_limit:
                    raise ValueError("Budget exceeded")
                if time.time() > deadline:
                    raise ValueError("Time exceeded")
                # Clip to box to satisfy constraints
                x_clipped = np.clip(x, -5.0, 5.0)
                return func(x_clipped)

            # Initial simplex: use the best point and small perturbations
            # Nelder-Mead requires dim+1 points
            # We can just let it generate the simplex, or provide one
            # Providing one is safer for reproducibility
            
            try:
                # maxiter is not the same as maxfun, but we control via wrapper
                # method='Nelder-Mead'
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': n_local, 'xatol': 1e-8, 'fatol': 1e-8})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                # If Nelder-Mead fails or exceeds budget/time, we keep the best from global phase
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
