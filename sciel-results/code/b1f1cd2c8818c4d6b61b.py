import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import warnings
warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        
        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        
        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0

        # Tracked function to count calls and update best
        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        bounds = [(-5.0, 5.0)] * self.dim

        # Phase 1: Global Exploration with Differential Evolution
        # DE is robust for Rosenbrock's narrow valley structure.
        # We tune it to use roughly half the budget to find the basin of attraction.
        # popsize=10 in 5D means a population of 50.
        # maxiter=5 means 5 generations. Total evals ~250.
        
        def de_callback(xk, convergence):
            # Stop if we are getting close to the budget limit
            return call_count >= self.budget - 50

        try:
            res = differential_evolution(
                tracked_func,
                bounds,
                maxiter=5,
                popsize=10,
                tol=1e-8,
                callback=de_callback,
                init='sobol',
                seed=42
            )
        except Exception:
            pass

        # Phase 2: Intensive Refinement with L-BFGS-B
        # L-BFGS-B converges very fast once in the basin.
        # We use the remaining budget for high-precision polishing.
        if time_ok() and call_count < self.budget - 5:
             try:
                 res = minimize(
                     tracked_func,
                     best_x,
                     method='L-BFGS-B',
                     bounds=bounds,
                     options={'maxiter': 1000, 'maxfun': self.budget - call_count - 5, 'ftol': 1e-15, 'gtol': 1e-10}
                 )
             except Exception:
                pass
        
        # Phase 3: Final Polish (if any budget left and time permits)
        # A quick 1+1-ES to ensure no precision is lost
        if time_ok() and call_count < self.budget:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.01
            remaining = self.budget - call_count
            
            for _ in range(remaining):
                if not time_ok() or call_count >= self.budget:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5.0, 5.0)
                fy = tracked_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                    
                if sigma < 1e-8:
                    break

        return best_x, best_fx
