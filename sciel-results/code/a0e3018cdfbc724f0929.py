"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead simplex optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin for 330s limit
        
        # State tracking
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        
        # We will use a wrapped function to count evaluations and enforce time limits
        eval_count = 0
        
        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError
            if eval_count >= self.budget:
                raise TimeoutError # Or return best known? No, raise to stop loop cleanly.
            
            # Ensure x is a valid vector
            x = np.asarray(x, dtype=float)
            if x.shape != (self.dim,):
                x = np.clip(x, -5, 5)
            
            fx = func(x)
            eval_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            return fx

        # Nelder-Mead is a local method. For Rosenbrock, starting near the origin 
        # or exploring a few distinct points is better than random uniform.
        # However, the minimizer is shifted. 
        # Strategy: Multi-start Nelder-Mead.
        # We have 500 evals. Nelder-Mead in 5D typically uses ~5 * iter evals.
        # Let's try a few starts.
        
        # Initial points: Origin, and a few others spread out.
        # Rosenbrock minimum is usually at (1,1,...,1) in standard form.
        # Here it's shifted. 
        # Let's start with the center of the box and a few others.
        
        starts = [
            np.zeros(self.dim),
            np.ones(self.dim),
            np.full(self.dim, -1.0),
            np.array([1.0, 1.0, 1.0, 1.0, 1.0]) # Standard Rosenbrock min location
        ]
        
        # Add a couple of random points if we have budget
        rng = np.random.default_rng(42)
        for _ in range(2):
            starts.append(rng.uniform(-4, 4, self.dim))

        for start_x in starts:
            if time.time() > deadline:
                break
            if eval_count >= self.budget:
                break
                
            # Configure Nelder-Mead
            # Initial simplex: start_x + small perturbation
            x0 = np.clip(start_x, -5, 5)
            
            # Options for scipy minimize
            # maxiter controls iterations, not function calls directly.
            # We rely on the wrapped_func raising an exception to stop.
            # However, scipy might catch exceptions? 
            # Better to use a callback to check time/budget and return True to stop.
            
            stop_flag = {'stop': False}
            
            def callback(xk):
                nonlocal eval_count
                if time.time() > deadline or eval_count >= self.budget:
                    stop_flag['stop'] = True
                    return True
                return False
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={
                        'xatol': 1e-6, 
                        'fatol': 1e-8,
                        'maxiter': 100, # Conservative, callback will stop earlier if needed
                        'adaptive': True
                    },
                    callback=callback
                )
                
                # Update best if scipy found a better one
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # If wrapped_func raised TimeoutError, we just break out
                break

        # If we haven't made any calls (e.g. exception in init?), we must make at least one?
        # The contract says "exception before the first call makes the whole result invalid".
        # Our wrapped_func is the first call. If it raises, we are invalid.
        # But we only raise if time > deadline or budget >= 500.
        # At start, time < deadline and budget < 500. So first call should succeed.
        
        # Ensure we return something
        return best_x, best_f
# EVOLVE-BLOCK-END
