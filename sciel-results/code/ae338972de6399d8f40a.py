"""Multi-Start Powell + Nelder-Mead Optimizer for Rosenbrock."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Global Search via Multi-Start Powell followed by Local Refinement via Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        # Use 280s limit to be safe within 300s
        self.deadline = self.start_time + 280.0

        def wrapped_func(x):
            # Check limits
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")

            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            
            # Update global best
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        # Phase 1: Multi-Start Powell
        # We will try a few starting points.
        # First, generate a small set of diverse initial points.
        # Using Latin Hypercube Sampling for better coverage.
        n_starts = 10
        # We need to evaluate these starts first.
        # But Powell itself will evaluate the start point.
        # To avoid double counting, we can just let Powell handle the start.
        
        # Let's pick 5 diverse points to start Powell from.
        # We use a fixed seed for reproducibility in the optimizer logic, 
        # but the function itself is seeded by the evaluator.
        rng = np.random.default_rng(42)
        
        # Generate 5 starting points in [-5, 5]^5
        starts = rng.uniform(-5, 5, size=(5, self.dim))
        
        # Sort starts by a quick heuristic? No, just run them.
        # We will run Powell on each start, but stop if we hit budget or time.
        # Powell is efficient.
        
        powell_results = []
        
        for i in range(5):
            if self.calls_made >= self.budget or time.time() > self.deadline:
                break
                
            start_x = starts[i]
            
            try:
                def powell_callback(xk):
                    return self.calls_made >= self.budget or time.time() > self.deadline

                res_powell = scipy.optimize.minimize(
                    wrapped_func,
                    start_x,
                    method='Powell',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-8,
                        'fatol': 1e-8
                    },
                    callback=powell_callback
                )
                
                if res_powell.fun < self.best_f:
                    self.best_f = res_powell.fun
                    self.best_x = np.clip(res_powell.x, -5, 5)
                
                powell_results.append(res_powell)
                
            except StopIteration:
                break
            except Exception:
                pass

        # Phase 2: Local Refinement using Nelder-Mead
        # Powell might leave some room for improvement, especially in the very narrow valley.
        # NM can polish.
        if self.best_x is not None and self.calls_made < self.budget:
            remaining_budget = self.budget - self.calls_made
            
            try:
                def nm_callback(xk):
                    return self.calls_made >= self.budget or time.time() > self.deadline

                res_nm = scipy.optimize.minimize(
                    wrapped_func,
                    self.best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500, # High limit, will be cut by budget
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    },
                    callback=nm_callback
                )
                
                if res_nm.fun < self.best_f:
                    self.best_f = res_nm.fun
                    self.best_x = np.clip(res_nm.x, -5, 5)
                    
            except StopIteration:
                pass
            except Exception:
                pass

        # If no calls were made (shouldn't happen), return initial guess
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = func(np.clip(self.best_x, -5, 5))

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
