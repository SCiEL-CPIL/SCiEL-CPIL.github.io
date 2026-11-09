"""Combined Differential Evolution + Nelder-Mead Optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Global Search via Differential Evolution followed by Local Refinement via Nelder-Mead."""

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

        # Define bounds for Differential Evolution
        bounds = [(-5.0, 5.0)] * self.dim

        # Phase 1: Global Search using Differential Evolution
        # We allocate ~60% of the budget to DE to explore the landscape.
        de_budget = int(self.budget * 0.6)
        
        # DE parameters:
        # popsize: factor for population size. popsize * dim.
        # For dim=5, popsize=2 gives 10 points per iteration.
        # 10 points/iter * 30 iters = 300 calls.
        # We want to stay within de_budget.
        # Let's estimate iterations: de_budget / (popsize * dim)
        # If popsize=2, dim=5, calls_per_iter=10.
        # max_iter = de_budget // 10
        
        # To be safe and adaptive, we can just let DE run and rely on the 
        # callback/wrapped_func to stop it if budget is hit.
        # However, DE doesn't have a direct "maxfev" option in older SciPy,
        # but it respects the function calls.
        
        # We need to be careful: DE initializes the population (popsize*dim calls)
        # and then iterates.
        
        # Let's use a smaller population to save calls for more iterations.
        # popsize=1 is invalid. popsize=2 is minimum.
        
        try:
            # Use a custom callback to stop DE if we hit budget or time
            def de_callback(xk, convergence):
                return self.calls_made >= de_budget or time.time() > self.deadline

            # differential_evolution does not have a callback parameter in all versions?
            # Actually, scipy.optimize.differential_evolution does NOT have a callback parameter.
            # It only has maxiter, tol, etc.
            # We must rely on the wrapped_func raising an exception or returning early?
            # If wrapped_func raises StopIteration, DE will crash.
            # We should catch the exception.
            
            # Alternative: Estimate maxiter.
            # Initial pop: popsize * dim calls.
            # Each iter: popsize * dim calls.
            # Total calls = (maxiter + 1) * popsize * dim.
            # We want Total calls <= de_budget.
            # maxiter = (de_budget // (popsize * dim)) - 1
            
            popsize = 2
            calls_per_iter = popsize * self.dim
            if calls_per_iter == 0:
                calls_per_iter = 1
            maxiter = max(1, (de_budget // calls_per_iter) - 1)
            
            res_de = scipy.optimize.differential_evolution(
                wrapped_func,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-8,
                seed=42,
                init='latinhypercube' # Better initial spread
            )
            
            if res_de.fun < self.best_f:
                self.best_f = res_de.fun
                self.best_x = np.clip(res_de.x, -5, 5)
                
        except StopIteration:
            pass
        except Exception:
            # If DE fails, we proceed with whatever best we have
            pass

        # Phase 2: Local Refinement using Nelder-Mead
        if self.best_x is not None and self.calls_made < self.budget:
            remaining_budget = self.budget - self.calls_made
            
            # Nelder-Mead is efficient. 
            # We can run it until budget is exhausted or it converges.
            # Nelder-Mead also doesn't have a direct maxfev in options, 
            # but 'maxiter' is available.
            # Each iteration of Nelder-Mead does ~dim+1 evaluations? 
            # Actually, it does a reflection, expansion, contraction, shrink.
            # Roughly 2-4 evaluations per iteration.
            # Let's set maxiter high and rely on budget check in wrapped_func.
            
            try:
                def nm_callback(xk):
                    return self.calls_made >= self.budget or time.time() > self.deadline

                res_nm = scipy.optimize.minimize(
                    wrapped_func,
                    self.best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000, # High limit, will be cut by budget
                        'xatol': 1e-8,
                        'fatol': 1e-8,
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
