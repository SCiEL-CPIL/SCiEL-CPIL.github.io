# EVOLVE-BLOCK-START
import numpy as np
import time
from cma import CMAEvolutionStrategy

class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    CMA-ES is highly effective for black-box continuous optimization, 
    especially for problems with non-linear dependencies and non-smoothness 
    like the BBOB f7 Step Ellipsoidal function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s hard limit. We use 290s to be safe.
        deadline = start_time + 290
        
        dim = self.dim
        budget = self.budget
        
        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None

        def wrapped(x):
            self._call_count += 1
            fx = float(func(x))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = np.array(x, dtype=float)
            return fx

        # Initialize CMA-ES
        # Start at center, initial sigma large enough to explore [-5, 5]^d
        x0 = np.zeros(dim)
        sigma0 = 2.0
        
        # Options for CMA-ES
        options = {
            'maxfevals': budget,
            'bounds': [(-5.0, 5.0)] * dim,
            'verbose': -9,
            'seed': 42,
            'CMA_diagonal': True, # Can help with step functions by decoupling dimensions initially
            'CMA_adaptation_of_diagonal': True
        }
        
        try:
            es = CMAEvolutionStrategy(x0, sigma0, options)
            
            while not es.stop() and self._call_count < budget:
                # Check time
                if time.time() > deadline:
                    break
                
                # Ask for next population
                # We need to be careful with the number of points asked vs budget remaining
                # CMA-ES asks for popsize at a time.
                # If remaining budget is less than popsize, we can still ask, 
                # but we must ensure we don't exceed budget.
                
                # However, standard CMA-ES wrapper in scipy/cma handles this by 
                # stopping when maxfevals is reached. 
                # But we have a strict call count limit.
                
                # To be safe, we can limit the population size if we are close to the end.
                # But usually, CMA-ES handles maxfevals correctly.
                # Let's just call ask and tell.
                
                # One issue: if we are at budget-1 and CMA-ES asks for 10 points, 
                # it will make 10 calls, exceeding budget.
                # We should check remaining budget.
                remaining = budget - self._call_count
                if remaining <= 0:
                    break
                
                # If remaining is small, we might just do random sampling or 
                # let CMA-ES handle it if it supports partial pop.
                # Standard cma library does not easily support partial population 
                # in the standard loop without hacking.
                
                # A safer approach for strict budget:
                # If remaining < popsize, we can set maxfevals to remaining?
                # No, maxfevals is total.
                
                # Let's just run the standard loop. The cma library stops 
                # when es.stop() is true, which includes maxfevals.
                # However, it might evaluate a full population even if it exceeds 
                # the limit by a few? No, it checks maxfevals.
                
                # Actually, the `cma` library's `es.ask()` returns a list of 
                # `popsize` points. If we are close to the limit, we might exceed it.
                # To strictly adhere to "More than 500 calls ... makes the whole result invalid",
                # we must ensure we never exceed 500.
                
                # Strategy:
                # 1. If remaining budget is small (e.g., < popsize), 
                #    we can manually sample `remaining` points from the current distribution?
                #    Or just stop CMA-ES and do random sampling?
                #    Or just stop?
                
                # Let's check the current popsize.
                popsize = es.popsize
                if remaining < popsize:
                    # We have less budget than one full population.
                    # We can ask for points, but only evaluate as many as we have budget for.
                    # However, CMA-ES tell() expects the full population results.
                    # If we don't provide all, it might error or behave unexpectedly.
                    
                    # Safer: Stop CMA-ES and use remaining budget for random sampling 
                    # around the current best or mean.
                    # Or, simply break and let the best found so far be the result.
                    # Given the step function, random sampling might not help much 
                    # if we are already close.
                    
                    # Let's try to evaluate only `remaining` points from the ask.
                    # But we need to handle the tell carefully.
                    
                    # Alternative: Just break. The CMA-ES has likely converged 
                    # or is near the end.
                    break
                
                # Ask for population
                xs = es.ask()
                
                # Evaluate each point
                fs = []
                for x in xs:
                    if self._call_count >= budget:
                        break
                    if time.time() > deadline:
                        break
                    fx = wrapped(x)
                    fs.append(fx)
                    
                # If we broke out early due to time/budget, we might have 
                # fewer fs than xs. 
                # CMA-ES tell() requires len(fs) == len(xs).
                # If we have a mismatch, we cannot call tell().
                # In that case, we just break.
                
                if len(fs) == len(xs):
                    es.tell(xs, fs)
                else:
                    break
                    
        except Exception:
            # If CMA-ES fails for any reason, fall back to simple random search
            # to ensure we use the remaining budget and don't crash.
            if self._call_count < budget:
                rng = np.random.default_rng(42)
                for _ in range(budget - self._call_count):
                    if time.time() > deadline:
                        break
                    x = rng.uniform(-5, 5, dim)
                    wrapped(x)

        # Ensure we return valid result
        if self._best_x is None:
            self._best_x = np.zeros(dim)
            # Make one final call if we haven't made any? 
            # But we must not exceed budget.
            if self._call_count == 0:
                self._best_f = func(self._best_x)
                self._call_count = 1
            else:
                # We have already made calls, so _best_x should be set.
                # If not, something went wrong, but we handle it above.
                pass

        return self._best_x, self._best_f


# EVOLVE-BLOCK-END
