import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 285.0
        calls_made = 0
        best_x = None
        best_f = np.inf
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                raise StopIteration
            if time.time() - start_time > time_limit:
                raise StopIteration
            calls_made += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        # Phase 1: Differential Evolution for Global Search
        # DE is robust to rotations and shifts.
        # Budget allocation: 40% for DE, 60% for L-BFGS-B
        de_budget = int(budget * 0.4)
        
        # Differential Evolution parameters
        # popsize is a factor. popsize=5 means 5*dim = 25 individuals.
        # With dim=5, 25 evals/gen. de_budget=200 -> ~8 generations.
        # This might be too few. Let's try popsize=3 (15 individuals).
        # 200 / 15 = 13 generations.
        
        try:
            # DE requires bounds
            bounds = [(-5.0, 5.0) for _ in range(dim)]
            
            # We use a custom function wrapper for DE
            def de_func(x_in):
                return safe_call(x_in)
                
            # maxiter is not directly budget-based in DE, but we can estimate
            # popsize=3 -> 15 individuals.
            # We want to use ~de_budget calls.
            # Initial population: 15 calls.
            # Each generation: 15 calls.
            # Total calls = 15 + 15 * generations.
            # 200 = 15 + 15 * g -> g = 12.3.
            
            res_de = differential_evolution(
                de_func, 
                bounds, 
                maxiter=15, 
                popsize=3,
                tol=1e-9,
                seed=0,
                polish=False  # We will do our own polish
            )
            
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = res_de.x.copy()
                
        except StopIteration:
            pass
        except Exception as e:
            pass

        # Phase 2: L-BFGS-B Polish
        if best_x is not None and calls_made < budget - 5 and time.time() - start_time < time_limit:
            try:
                def lbfun(x_in):
                    return safe_call(x_in)
                
                bounds = [(-5.0, 5.0) for _ in range(dim)]
                
                res_lbfgs = minimize(
                    lbfun, 
                    best_x, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'maxfun': budget - calls_made, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res_lbfgs.fun < best_f:
                    best_f = res_lbfgs.fun
                    best_x = res_lbfgs.x.copy()
                    
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 3: If L-BFGS-B didn't improve much, try a second polish from a slightly perturbed point
        if best_x is not None and calls_made < budget - 5 and time.time() - start_time < time_limit:
            # Check if we are close to a flat region or stuck
            # Just try one more L-BFGS-B from a random neighbor if we have budget
            if best_f > 1e-5: # If not at machine precision
                rng = np.random.default_rng(1)
                x2 = best_x + 0.1 * rng.standard_normal(dim)
                x2 = np.clip(x2, -5.0, 5.0)
                try:
                    f2 = safe_call(x2)
                    if f2 < best_f:
                        best_f = f2
                        best_x = x2.copy()
                        
                    def lbfun2(x_in):
                        return safe_call(x_in)
                        
                    res_lbfgs2 = minimize(
                        lbfun2, 
                        best_x, 
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 50, 'maxfun': budget - calls_made}
                    )
                    if res_lbfgs2.fun < best_f:
                        best_f = res_lbfgs2.fun
                        best_x = res_lbfgs2.x.copy()
                except StopIteration:
                    pass
                except Exception:
                    pass

        return best_x, best_f
