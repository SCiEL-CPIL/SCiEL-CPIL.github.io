"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import cma

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        # CMA-ES parameters
        # For dim=5, population size default is 4 + 3*ln(5) ≈ 9.
        # We can tune it slightly. 
        # Initial sigma: The box is [-5, 5]. Center is 0.
        # A sigma of 2.0 covers a large portion of the space initially.
        x0 = np.zeros(self.dim)
        sigma0 = 2.0
        
        # Options
        opts = {
            'maxiter': 1000,  # High enough, we will stop based on time/budget
            'verbose': -9,    # Silent
            'seed': 42,
            'CMA_boundaries': [(-5.0, 5.0)] * self.dim,
            'CMA_boundaryhandler': 'transform', # Maps to box
            'CMA_adapt_sigma': 'default',
            'CMA_elitism': True, # Helps with Rosenbrock
            'CMA_popsize': 12,  # Slightly larger pop for better exploration
            'CMA_lambdastar': 1, # Elite count
        }
        
        # Wrap function to count calls and check time
        call_count = 0
        best_x = x0.copy()
        best_f = float('inf')
        
        def counted_func(x):
            nonlocal call_count, best_x, best_f
            call_count += 1
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val
        
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        
        try:
            # Loop until termination condition or time/budget limit
            while not es.stop() and call_count < self.budget:
                if time.time() > deadline:
                    break
                
                # Ask for candidates
                X = es.ask()
                
                # Evaluate
                fs = []
                for x in X:
                    if time.time() > deadline:
                        break
                    f_val = counted_func(x)
                    fs.append(f_val)
                
                # If we broke early, we might have fewer fs than X
                # CMA-ES expects a list of same length as X
                # If we broke, we need to handle this carefully or just break the loop
                if len(fs) < len(X):
                     # Fill remaining with inf? Or just break.
                     # Breaking is safer to avoid corrupting the CMA-ES state with bad data.
                     break
                
                es.tell(X, fs)
                
        except Exception:
            # Time limit or other error. We keep the best found so far.
            pass

        return best_x, best_f

# EVOLVE-BLOCK-END
