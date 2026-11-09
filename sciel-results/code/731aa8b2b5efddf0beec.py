"""Hybrid Optimizer: Sobol + Nelder-Mead for Bent Cigar."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    """
    Hybrid Optimizer: Sobol Sequence Sampling + Nelder-Mead Refinement.
    Designed for anisotropic functions like Bent Cigar.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State
        best_x = None
        best_fx = np.inf
        calls_used = 0
        
        def evaluate(x):
            nonlocal best_x, best_fx, calls_used
            if calls_used >= budget or time.time() >= deadline:
                return None
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_used += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Coarse Search using Sobol Sequence
        # We allocate ~60% of the budget for coarse search
        coarse_budget = int(budget * 0.6)
        
        # Generate Sobol points
        # We need enough points to fill the coarse budget
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        
        # Generate exactly coarse_budget points
        # Note: qmc.Sobol.random(n) generates n points
        sobol_points = sampler.random(coarse_budget)
        
        # Map [0, 1]^d to [-5, 5]^d
        sobol_points_mapped = sobol_points * 10.0 - 5.0
        
        # Evaluate Sobol points
        for x in sobol_points_mapped:
            if calls_used >= coarse_budget or time.time() >= deadline:
                break
            evaluate(x)
            
        # Phase 2: Local Refinement using Nelder-Mead
        # Use the best point found so far as the initial simplex center.
        if best_x is not None:
            # Define the objective function for minimize
            def objective(x):
                nonlocal calls_used, best_fx, best_x
                if calls_used >= budget or time.time() >= deadline:
                    # Raise exception to stop minimize
                    raise ValueError("Budget or time limit exceeded")
                
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                calls_used += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x_clipped.copy()
                return fx

            # Nelder-Mead options
            # We have budget - calls_used left.
            
            x0 = best_x.copy()
            
            try:
                res = minimize(
                    objective, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000, # High limit, controlled by budget/exception
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass # If it fails, we keep the best from Phase 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
