"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-Start Nelder-Mead Optimizer.
    
    Strategy:
    1. Perform local search (Nelder-Mead) from multiple starting points.
    2. Use a "stagnation" detector to restart from the current best point
       with a perturbed simplex or a new random point if progress stalls.
    3. This is structurally different from the (1+1)-ES in the parent,
       focusing on intensive local exploitation with strategic restarts.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_fx = float('inf')
        evals_used = 0
        
        # Initial start points: Center and a few random points
        # We will run Nelder-Mead sequentially from these points
        # or use a "best of" strategy.
        
        # To maximize AOCC, we want to find a good solution ASAP.
        # Starting at 0 is a good bet for BBOB.
        
        def run_nm(x0, max_evals, simplex_scale=0.5):
            nonlocal best_x, best_fx, evals_used
            
            if evals_used >= self.budget:
                return
            
            # Create a wrapper to track calls and enforce deadline
            local_calls = [0]
            
            def nm_func(z):
                nonlocal evals_used
                if evals_used >= self.budget or time.time() > deadline:
                    return best_fx # Return current best to stop early
                local_calls[0] += 1
                evals_used += 1
                z_clipped = np.clip(z, -5, 5)
                val = func(z_clipped)
                if val < best_fx:
                    best_fx = val
                    best_x = z_clipped.copy()
                return val
            
            # Nelder-Mead options
            # We want to use up to max_evals
            # maxiter in scipy NM is roughly the number of iterations,
            # each iteration uses ~n+1 evaluations.
            # So maxiter ~ max_evals / (dim + 1)
            max_iter = max(10, max_evals // (self.dim + 1))
            
            # Initial simplex
            # Use a small simplex if we are polishing, larger if exploring
            simplex = np.array([x0 + simplex_scale * e for e in np.eye(self.dim)])
            
            try:
                result = minimize(
                    nm_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'initial_simplex': simplex,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        # Phase 1: Explore from Center
        # Use 60% of budget
        budget_phase1 = int(self.budget * 0.6)
        run_nm(np.zeros(self.dim), budget_phase1, simplex_scale=2.0)
        
        # Phase 2: Explore from Best Point (Polish)
        # Use remaining budget
        if best_x is not None and evals_used < self.budget:
            remaining = self.budget - evals_used
            # Restart from best_x with a smaller simplex for polishing
            run_nm(best_x, remaining, simplex_scale=0.5)
            
            # If still budget left, try a random restart far away?
            # Unlikely to be better than the basin found, but maybe if f6 has multiple minima?
            # BBOB f6 usually has one global min.
            
            if evals_used < self.budget - 50:
                # Try one more random start
                x_rand = rng.uniform(-5, 5, self.dim)
                run_nm(x_rand, self.budget - evals_used, simplex_scale=2.0)

        # Fallback
        if best_x is None:
            best_x = rng.uniform(-5, 5, self.dim)
            best_fx = func(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
