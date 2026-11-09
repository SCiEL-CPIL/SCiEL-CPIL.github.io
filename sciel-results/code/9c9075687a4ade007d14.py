import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        
        # Initial guess
        x0 = np.zeros(self.dim)
        best_x = x0.copy()
        best_f = func(best_x)
        call_count = 1
        
        def wrapped_func(x):
            nonlocal best_x, best_f, call_count
            # Check time
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            # Enforce budget
            if call_count >= self.budget:
                raise StopIteration("Budget exceeded")
            
            # Ensure x is a 1D numpy array
            x = np.asarray(x).flatten()
            
            fx = func(x)
            call_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            return fx

        try:
            # Phase 1: Global Exploration with Differential Evolution
            # Use ~20% of budget for global search
            de_budget = 100
            de_res = scipy.optimize.differential_evolution(
                wrapped_func,
                bounds=[(-5, 5) for _ in range(self.dim)],
                maxiter=10,
                popsize=10,
                tol=1e-8,
                seed=42,
                init='latinhypercube'
            )
            # Update best if DE found something better
            # (wrapped_func already updates best_x, best_f, so this is redundant but safe)
            
            # Phase 2: Local Exploitation with Powell's Method
            # Use remaining budget
            remaining_budget = self.budget - call_count
            if remaining_budget > 10:
                powell_res = scipy.optimize.minimize(
                    wrapped_func,
                    best_x,
                    method='Powell',
                    bounds=[(-5, 5) for _ in range(self.dim)],
                    options={'maxfev': remaining_budget, 'xtol': 1e-8, 'ftol': 1e-8}
                )
                
        except TimeoutError:
            pass
        except StopIteration:
            pass
        except Exception:
            pass

        return best_x, best_f
