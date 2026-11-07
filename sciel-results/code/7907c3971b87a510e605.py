"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """Hybrid Differential Evolution and Local Optimization for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        self.calls = 0
        budget = self.budget
        dim = self.dim
        
        class StopSearch(Exception):
            pass

        def wrapped_func(x):
            self.calls += 1
            if self.calls > budget:
                raise StopSearch()
            return float(func(x))

        bounds = [(-5, 5)] * dim
        
        best_x = None
        best_f = np.inf
        
        # Phase 1: Global Exploration using Differential Evolution
        # DE uses a population-based approach which is better for multimodal functions.
        # popsize=5, dim=5 -> population size 25.
        # maxiter=10 -> approx 250 evaluations.
        try:
            de_result = differential_evolution(
                wrapped_func, 
                bounds, 
                maxiter=10, 
                popsize=5, 
                tol=1e-8, 
                atol=1e-8,
                seed=42,
                init='latinhypercube'
            )
            
            if de_result.fun < best_f:
                best_f = de_result.fun
                best_x = de_result.x.copy()
                
        except StopSearch:
            pass
        except Exception:
            pass

        # Phase 2: Local Refinement
        # Use the best point from DE to start a local search.
        if best_x is not None:
            # Run Nelder-Mead for refinement
            try:
                nm_result = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if nm_result.fun < best_f:
                    best_f = nm_result.fun
                    best_x = nm_result.x.copy()
            except StopSearch:
                pass
            except Exception:
                pass
            
            # Try Powell as well if budget remains
            if self.calls < budget - 10:
                try:
                    powell_result = minimize(
                        wrapped_func, 
                        best_x, 
                        method='Powell', 
                        options={'maxiter': 100, 'xtol': 1e-8, 'ftol': 1e-10}
                    )
                    if powell_result.fun < best_f:
                        best_f = powell_result.fun
                        best_x = powell_result.x.copy()
                except StopSearch:
                    pass
                except Exception:
                    pass
        
        # Fallback: If best_x is still None, do a simple random search
        if best_x is None:
            for _ in range(10):
                if self.calls >= budget:
                    break
                x = np.random.uniform(-5, 5, dim)
                try:
                    f_val = wrapped_func(x)
                    if f_val < best_f:
                        best_f = f_val
                        best_x = x.copy()
                except StopSearch:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
