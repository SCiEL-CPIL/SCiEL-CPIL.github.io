import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None
        self.best_x = None
        self.best_f = float('inf')

    def _evaluate(self, x):
        if self.calls >= self.budget:
            return None
        self.calls += 1
        val = self.func(x)
        if val < self.best_f:
            self.best_f = val
            self.best_x = x.copy()
        return val

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        d = self.dim
        self.func = func
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = None
        
        # Strategy:
        # 1. Multi-start L-BFGS-B to handle hidden shifts/rotations and find the basin.
        #    Bent Cigar is unimodal but ill-conditioned. L-BFGS-B with bounds is effective
        #    if started reasonably. Multi-start increases robustness.
        # 2. Nelder-Mead for final polishing. NM is derivative-free and can handle
        #    the final convergence where L-BFGS-B might stall due to ill-conditioning.
        
        def wrapper_func(x, deadline):
            if self.calls >= self.budget or time.time() > deadline - 2:
                return 1e10
            f_val = self._evaluate(x)
            if f_val is None:
                return 1e10
            return f_val

        # Phase 1: Multi-start L-BFGS-B
        # Generate a few starting points. Origin is usually good, but random points
        # within the box might help if the shift is significant.
        
        # Start with origin
        x0 = np.zeros(d)
        f0 = self._evaluate(x0)
        if f0 is None:
            return np.zeros(d), 0.0

        # Run L-BFGS-B from origin
        try:
            res = minimize(wrapper_func, x0, method='L-BFGS-B', 
                           bounds=[(-5, 5)] * d, 
                           args=(deadline,),
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            pass

        # Check budget and time
        if self.calls < self.budget - 50 and time.time() < deadline - 20:
            # Try a few random starts if we have budget
            rng = np.random.default_rng(42)
            for _ in range(3):
                if self.calls >= self.budget - 50 or time.time() > deadline - 10:
                    break
                
                # Random point within [-3, 3] to stay safely inside bounds
                x_start = rng.uniform(-3, 3, d)
                
                # If this point is better than current best, it's worth optimizing from
                # But we don't know f(x_start) without evaluating. 
                # L-BFGS-B will evaluate it. 
                # To avoid wasting calls on bad starts, we could do a quick check,
                # but L-BFGS-B is fast. Let's just run it.
                
                try:
                    res2 = minimize(wrapper_func, x_start, method='L-BFGS-B', 
                                    bounds=[(-5, 5)] * d, 
                                    args=(deadline,),
                                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
                except Exception:
                    pass

        # Phase 2: Nelder-Mead Polish
        # Use NM to refine the best point found so far.
        # NM is good for final convergence on smooth functions.
        
        if self.calls < self.budget - 10 and time.time() < deadline - 10:
            if self.best_x is None:
                self.best_x = x0.copy()
            
            try:
                res3 = minimize(wrapper_func, self.best_x, method='Nelder-Mead',
                                args=(deadline,),
                                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-12})
            except Exception:
                pass
        
        # Return the best found
        if self.best_x is None:
            self.best_x = np.zeros(d)
            self.best_f = 0.0
            
        return self.best_x, self.best_f
