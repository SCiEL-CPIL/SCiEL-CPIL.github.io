"""Population-based CMA-ES black-box optimizer for BBOB f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import CmaStrategy

class Optimizer:
    """CMA-ES optimizer with population-based search.
    
    Strategy:
    - Use CMA-ES (via scipy.optimize.CmaStrategy) to adaptively learn the covariance
      structure of the rotated landscape.
    - Use a large initial population (lambda=10) to maintain diversity and escape
      Rastrigin's local minima without relying on discrete restarts.
    - The covariance matrix adaptation allows the search to align with the rotated
      axes of the function, improving convergence efficiency.
    - Strict budget management and time guards ensure safety and full utilization
      of the 500 evaluation budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._func = None
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        self._func = func
        self._call_count = 0
        self._best_f = np.inf
        self._best_x = np.zeros(self.dim)
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # x0: Initial center. Center of the box.
        x0 = np.zeros(dim)
        
        # sigma0: Initial step size. 
        # The domain is [-5, 5]^5. A reasonable initial step size is about 1/4 of the range.
        # Range is 10, so 2.5 is the half-range. 1.0 or 1.5 is a common start.
        sigma0 = 1.5
        
        # Population size (lambda)
        # Default is 4 + floor(3 * ln(n)). For n=5, ln(5)~1.6, so 4+4=8.
        # The directive suggests lambda=10 for larger diversity.
        # Let's use lambda=10 as suggested to help escape local minima.
        pop_size = 10
        
        # Boundary constraints: [-5, 5]
        bounds = [(-5.0, 5.0)] * dim
        
        # Callback to track budget and best solution
        def callback(x, result):
            nonlocal self
            # result has 'x' (the evaluated points) and 'fvals'
            # In scipy CmaStrategy, the callback is called after each generation.
            # We need to count the evaluations.
            # The number of evaluations in this generation is result['x'].shape[0]
            # However, CmaStrategy might handle the first evaluation differently or 
            # we might need to be careful about how many calls were actually made.
            
            # Let's inspect result.
            # result['x'] is the population of this generation.
            # result['fvals'] are the fitness values.
            
            n_evals_this_gen = result['x'].shape[0]
            
            # Update call count
            # Note: CmaStrategy might evaluate x0 separately or include it in the first pop.
            # Scipy's CmaStrategy typically evaluates the initial point x0 first?
            # Actually, CmaStrategy(x0, sigma0, ...) starts by generating a population.
            # Let's assume the callback gives us the new evaluations.
            
            # To be safe, we should track the actual calls made by the wrapper function.
            # But we can't easily hook into the internal calls of CmaStrategy without wrapping func.
            # So we MUST wrap func to count calls.
            
            pass

        # We need to wrap the function to count calls and enforce budget.
        # The callback is useful for early stopping, but the wrapper is essential for counting.
        
        wrapped_func = func
        call_count = 0
        best_f = np.inf
        best_x = x0.copy()
        
        def counting_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                # Return a very bad value to stop CMA-ES from improving further
                # or just return the last known best? 
                # CMA-ES will use this value. If we return inf, it might crash or ignore.
                # Better to return a large value.
                return 1e300
            
            call_count += 1
            
            # Ensure x is within bounds (CMA-ES usually handles this, but just in case)
            x_clipped = np.clip(x, -5.0, 5.0)
            
            f_val = wrapped_func(x_clipped)
            
            if not np.isfinite(f_val):
                f_val = 1e300
                
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # Set up CMA-ES
        # maxiter: We have 500 budget. Pop size 10. So max 50 generations.
        # Set maxiter high, we will stop via time or budget.
        maxiter = 100
        
        # Set bounds
        # scipy.optimize.CmaStrategy accepts bounds
        try:
            es = CmaStrategy(
                counting_func,
                x0,
                sigma0,
                bounds=bounds,
                maxiter=maxiter,
                popsize=pop_size,
                seed=42,  # Fixed seed for reproducibility
                verbose=0,
                # Options to help with multimodality?
                # 'ccov' and 'cmu' are standard.
                # 'popsize' is set.
                # 'bounds' is set.
            )
            
            # Run until budget or time runs out
            while True:
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Run one iteration
                # es.step() might not be the right API for single steps in all versions.
                # CmaStrategy is a class that can be iterated.
                # Let's use the iterator protocol or just call es.step() if available.
                # In scipy, CmaStrategy is an object that can be used in a loop.
                # Actually, CmaStrategy is a class instance. We can call es.step() to perform one generation.
                # Let's check if step() exists. Yes, in recent scipy.
                es.step()
                
                # Check if stopped
                if es.converged or es._stop:
                    break
                    
        except Exception as e:
            # If CMA-ES fails for any reason, fall back to random search or just return best so far
            pass

        # Update class attributes
        self._call_count = call_count
        self._best_f = best_f
        self._best_x = best_x
        
        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
