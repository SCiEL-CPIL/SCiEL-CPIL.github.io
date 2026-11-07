import numpy as np
import time
from cma import CMA, CMAOptions

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin

        best_f = np.inf
        best_x = np.zeros(dim)
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            try:
                f = func(x)
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except:
                return np.inf

        # CMA-ES configuration
        # Popsize: 5 is a good balance for dim=5 (4+3*ln(5) ~ 8.7, but 5 is faster and often enough for 5D)
        # We can try 5 or 6. Let's use 5 to maximize iterations within budget.
        popsize = 5
        max_iter = budget // popsize + 10

        # Initial sigma: The box is [-5, 5]. A reasonable initial step is 1.0 or 2.0.
        # Given the "Sharp ridge" and hidden shift/rotation, starting broad is good.
        x0 = np.zeros(dim)
        sigma0 = 2.0

        # Options
        opts = {
            'maxiter': max_iter,
            'verbose': -9,  # Silence output
            'seed': 12345,  # Deterministic
            'bounds': [(-5.0) * np.ones(dim), 5.0 * np.ones(dim)],
            'tolx': 1e-6,
            'tolfun': 1e-8,
            'stopiters': 10, # Stop if no improvement for 10 iterations
        }

        try:
            es = CMA(x0, sigma0, opts)
            
            while not es.should_stop():
                if time.time() > deadline:
                    break
                
                # Ask for a batch of points
                solutions = es.ask()
                
                for x in solutions:
                    f = safe_call(x)
                    if time.time() > deadline:
                        break
                    es.tell(x, f)
                    
                # Check if we should stop early due to budget or time
                if calls_made >= budget or time.time() > deadline:
                    break
                    
        except Exception as e:
            # If CMA fails, we just return the best found so far
            pass

        return best_x, best_f
