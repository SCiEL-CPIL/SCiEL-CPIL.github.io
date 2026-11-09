"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer using a two-phase strategy:
    1. Global exploration using a deterministic Sobol sequence for maximal coverage.
    2. Local refinement using Nelder-Mead on the best point found.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit. Use 320s to be safe.
        deadline = start_time + 320.0
        
        dim = self.dim
        lo, hi = -5.0, 5.0
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = float('inf')

        def call_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return float('inf')
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Exploration with Sobol Sequence
        # Use the first 50% of the budget for global exploration.
        # Sobol sequences provide low-discrepancy points, ensuring good coverage.
        n_global = budget // 2
        
        # Generate Sobol points in [0, 1]^dim
        # scipy.stats.qmc.Sobol is available in recent scipy versions.
        # If not available, we can fall back to random or a simple grid, 
        # but assuming standard scipy is present.
        try:
            from scipy.stats.qmc import Sobol
            # Sobol requires power of 2 for sample size for best properties, 
            # but it works for any size.
            sampler = Sobol(d=dim, scramble=True, seed=42)
            # We want n_global points.
            # Sobol.generate takes a number of points.
            # Note: Sobol.generate returns points in [0,1].
            points = sampler.generate(n_global)
            
            for i in range(n_global):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                    
                x_unit = points[i]
                x = x_unit * (hi - lo) + lo
                call_func(x)
                
        except ImportError:
            # Fallback if Sobol is not available (unlikely in modern scipy)
            rng = np.random.default_rng(42)
            for i in range(n_global):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                x = rng.uniform(lo, hi, dim)
                call_func(x)

        if best_x is None:
            # Fallback if no calls were made (shouldn't happen)
            best_x = np.array([0.0] * dim)
            best_f = func(best_x)
            calls_made += 1

        # Phase 2: Local Refinement
        # Use Nelder-Mead from the best point found.
        # Nelder-Mead is derivative-free and works well in low dimensions.
        # We need to wrap the function to count calls and respect the deadline.
        
        if calls_made < budget and time.time() < deadline:
            # Set up Nelder-Mead options
            # maxiter is not the limit, maxfev is.
            # We have budget - calls_made evaluations left.
            # However, Nelder-Mead might use more evaluations per iteration.
            # Let's reserve a portion of the remaining budget.
            remaining = budget - calls_made
            
            # We'll allow Nelder-Mead to use up to 'remaining' function evaluations.
            # But we must stop if time runs out.
            
            def wrapper(x):
                nonlocal calls_made
                if time.time() > deadline:
                    raise Exception("Time limit reached")
                return call_func(x)

            try:
                # Nelder-Mead parameters
                # initial_simplex is auto-generated if not provided.
                # We can provide a simple initial simplex around best_x.
                # Default step size is 0.05 * x or 1.0 if x is 0.
                # Let's set a reasonable initial step size.
                res = minimize(
                    wrapper, 
                    best_x, 
                    method='Nelder-Mead',
                    options={
                        'maxfev': remaining,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # If Nelder-Mead fails or times out, we just keep the best from Phase 1
                pass

        # If we still have budget and time, we could do another local refinement
        # or a few random perturbations around the best point to escape potential local minima.
        # Given the Schwefel function is multimodal, Nelder-Mead might get stuck.
        # However, Phase 1 should have found a decent basin.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
