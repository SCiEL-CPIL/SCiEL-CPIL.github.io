"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer with structured initialization.
    
    Structural change: Replaces global DE search with diverse initial sampling
    followed by multiple local Nelder-Mead refinements. This exploits the
    basin-of-attraction structure of Büche-Rastrigin where local search is
    highly effective once in the correct basin.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        global_best_x = None
        global_best_fx = np.inf
        calls_made = 0

        def try_update(x, fx):
            nonlocal global_best_x, global_best_fx
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()

        def evaluate(x):
            nonlocal calls_made
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            try_update(x, fx)
            return fx

        # Phase 1: Structured Initialization (Latin Hypercube-like)
        # We want diverse coverage of [-5, 5]^dim.
        # Let's use 25 initial points to ensure good coverage in 5D.
        n_init = 25
        init_points = []
        
        # Use a stratified sampling approach to ensure spread
        # For each dimension, we divide the range into n_init bins and pick one point per bin.
        # This is a simple LHS approximation.
        for i in range(n_init):
            point = np.zeros(dim)
            for d in range(dim):
                # Bin i in dimension d
                bin_start = -5 + (10 * i / n_init)
                bin_end = -5 + (10 * (i + 1) / n_init)
                point[d] = rng.uniform(bin_start, bin_end)
            init_points.append(point)
            
        # Also add the center and some random points for robustness
        init_points.append(np.zeros(dim))
        for _ in range(5):
            init_points.append(rng.uniform(-5, 5, size=dim))
            
        # Evaluate initial points
        init_fits = []
        for p in init_points:
            if calls_made >= budget or time.time() > deadline:
                break
            fx = evaluate(p)
            if fx != np.inf:
                init_fits.append((fx, p))
                
        if not init_fits:
            # Fallback if something went wrong
            return global_best_x, global_best_fx

        # Sort by fitness
        init_fits.sort(key=lambda x: x[0])
        
        # Phase 2: Multi-start Local Refinement
        # Select top K points to refine
        # We have roughly budget - n_init - 5 calls left.
        # Nelder-Mead in 5D typically needs ~50-100 calls to converge well.
        # Let's allocate budget for 5 local searches.
        n_refines = 5
        remaining_budget = budget - calls_made
        
        if remaining_budget <= 0:
            return global_best_x, global_best_fx
        
        # Distribute remaining budget among refines
        # Give more budget to better starting points? Or equal?
        # Equal is safer.
        budget_per_refine = remaining_budget // n_refines
        
        # If budget per refine is too small, reduce number of refines
        if budget_per_refine < 10:
            n_refines = max(1, remaining_budget // 10)
            budget_per_refine = remaining_budget // n_refines

        for i in range(min(n_refines, len(init_fits))):
            if calls_made >= budget or time.time() > deadline:
                break
            
            start_x, start_fx = init_fits[i]
            
            # Run Nelder-Mead
            # maxiter is a limit on iterations, not function calls.
            # In 5D, one iteration is 6 function calls (initial simplex + 5 replacements/expansions/contractions).
            # So maxiter = budget_per_refine // 6 is a safe estimate.
            max_iter = max(1, budget_per_refine // 6)
            
            try:
                res = minimize(
                    func, 
                    start_x, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter, 
                        'xatol': 1e-4, 
                        'fatol': 1e-6,
                        'adaptive': True # Helps with scaling
                    }
                )
                # Update global best if this run found something better
                if res.fun < global_best_fx:
                    global_best_fx = res.fun
                    global_best_x = res.x.copy()
            except Exception:
                pass

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
