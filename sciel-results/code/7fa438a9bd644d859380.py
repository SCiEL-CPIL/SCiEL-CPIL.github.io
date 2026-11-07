import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, keep 30s buffer for safety
        time_limit = 300 - 30
        
        calls_made = 0
        best_x = np.random.default_rng(42).uniform(-5, 5, self.dim)
        best_f = float('inf')
        
        def wrapper(x):
            nonlocal calls_made, best_x, best_f
            # Time check: Raise exception to abort SciPy loops immediately
            if time.time() - start_time > time_limit:
                raise TimeoutError("Time limit exceeded")
            # Budget check
            if calls_made >= self.budget:
                raise RuntimeError("Budget exceeded")
            
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                return best_f
            
            if np.isnan(f_val) or np.isinf(f_val):
                return best_f

            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val

        # Phase 1: Global Search with Differential Evolution
        # Use ~60% of budget for DE to ensure good basin discovery
        # popsize=15, maxiter=20 -> max 300 calls (plus initial pop)
        try:
            res_de = differential_evolution(
                wrapper, 
                bounds=[(-5, 5)] * self.dim, 
                popsize=15, 
                maxiter=20,
                seed=42,
                tol=1e-6,
                init='latinhypercube'
            )
            # DE returns the best found, but our wrapper tracks it too
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = np.array(res_de.x)
        except (TimeoutError, RuntimeError):
            pass
        except Exception:
            pass

        # Phase 2: Local Polish with L-BFGS-B
        # Use remaining budget for high-precision convergence
        # L-BFGS-B is efficient for smooth functions like Rosenbrock
        if calls_made < self.budget and (time.time() - start_time) < time_limit:
            try:
                res_lbfgsb = minimize(
                    wrapper, 
                    best_x, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'gtol': 1e-12, 'ftol': 1e-12}
                )
                if res_lbfgsb.fun < best_f:
                    best_f = res_lbfgsb.fun
                    best_x = np.array(res_lbfgsb.x)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        # Phase 3: Final Nelder-Mead Polish (optional, if budget remains)
        # Non-gradient method for final robustness
        if calls_made < self.budget and (time.time() - start_time) < time_limit:
            try:
                res_nm = minimize(
                    wrapper, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = np.array(res_nm.x)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        return best_x, best_f
