"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import cma

class Optimizer:
    """CMA-ES based optimizer for BBOB f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Safety margin for time
        start_time = time.time()
        deadline = start_time + 290.0
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        # Wrapper to count calls and handle time
        def wrapped_func(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                return best_f
            if calls_made >= budget:
                return best_f
            
            # CMA-ES might propose points outside bounds.
            # The task says func clips, but we should also ensure our internal state 
            # doesn't drift too far if we were to use gradient, but here we just pass to func.
            # However, returning a value for out-of-bounds points to CMA-ES is tricky.
            # The task says: "func(x) clips a finite length-5 vector to the box and returns a float."
            # So we can just pass x to func. The clipping happens inside func.
            # But CMA-ES works better if we don't feed it infinite values or if we handle bounds.
            # Since func clips, out-of-bounds points will be evaluated at the boundary.
            # This is a valid strategy (penalty-free, just clipping).
            
            x = np.asarray(x)
            f_val = func(x)
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        # CMA-ES setup
        # Initial mean: center of the box
        x0 = np.zeros(dim)
        
        # Initial sigma: 
        # The box is [-5, 5]^5. Width is 10.
        # A sigma of 1.0 or 2.0 is a reasonable start to explore the space.
        # Given dim=5, a sigma of 1.0 is standard.
        sigma0 = 1.0
        
        # Population size
        # Default is 4 + floor(3 * ln(n)) = 4 + 3*ln(5) ~ 4 + 4.8 = 8.
        # We have 500 budget. 500 / 8 = 62.5 generations.
        # This is sufficient for CMA-ES to converge in 5D.
        
        options = {
            'maxfevals': budget, 
            'seed': 42, 
            'verbose': -9, 
            'bounds': [-5, 5], 
            'tolfun': 1e-8, 
            'tolx': 1e-8,
            'stopiters': 200 # Safety cap on iterations
        }
        
        es = cma.CMAEvolutionStrategy(x0, sigma0, options)
        
        while not es.stop() and calls_made < budget and time.time() < deadline:
            solutions = es.ask()
            for x in solutions:
                if time.time() > deadline or calls_made >= budget:
                    break
                f_val = wrapped_func(x)
                es.tell(x, f_val)
                
        # After CMA-ES finishes, we might have budget left.
        # We can try a simple local polish if time permits, but CMA-ES is usually quite good.
        # Let's just return the best found.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
