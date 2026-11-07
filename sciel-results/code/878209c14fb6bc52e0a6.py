"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using Differential Evolution + Powell."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer using a two-phase strategy:
    1. Global Phase: Differential Evolution (DE) to find the global basin.
       DE is robust to multimodality and rotation. We limit its evaluations to ~40% of budget.
    2. Local Phase: Powell's method starting from the best DE point.
       Powell is derivative-free and often faster/converges better than Nelder-Mead 
       for smooth-ish functions after a global search has narrowed the basin.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        best_x = None
        fbest = np.inf
        calls = 0

        def update_best(x, fx):
            nonlocal best_x, fbest
            if fx < fbest:
                fbest = fx
                best_x = x.copy()

        # --- Phase 1: Global Search via Differential Evolution ---
        # DE is a population-based stochastic method. 
        # For dim=5, a population size of 15 is reasonable.
        # We allocate ~200 evaluations to DE (40% of 500).
        n_de_budget = min(200, budget // 2)
        
        # Bounds for DE
        bounds = [(-5.0, 5.0)] * dim
        
        # We need a wrapper to count calls and enforce limits strictly.
        def de_objective(x):
            nonlocal calls, best_x, fbest
            if not time_ok():
                raise RuntimeError("Time limit exceeded")
            if calls >= budget:
                raise RuntimeError("Budget exceeded")
            
            calls += 1
            fx = func(x)
            update_best(x, fx)
            return fx

        # Run DE
        # maxiter is the number of generations. 
        # Popsize default is 15*dim = 75. 
        # Total calls approx = popsize * (1 + maxiter).
        # We want ~200 calls. 200 / 75 is ~2.6 generations. That's too few.
        # Let's use a smaller popsize to get more generations within the call budget.
        # Popsize = 10. Calls = 10 * (1 + maxiter). 
        # If maxiter=19, calls = 10 * 20 = 200.
        
        try:
            # tol and xatol set to high values so it runs until maxiter or our exception
            # We rely on the exception in de_objective to stop it if budget is hit.
            res = minimize(
                de_objective,
                np.zeros(dim), # Initial guess, DE will ignore it mostly
                method='differential_evolution',
                bounds=bounds,
                options={
                    'popsize': 10,
                    'maxiter': 20, # Approx 200 calls
                    'tol': 1.0,
                    'xatol': 1.0,
                    'init': 'latinhypercube',
                    'seed': 42
                }
            )
        except RuntimeError:
            pass
        except Exception:
            pass

        if best_x is None:
            # Fallback if DE failed completely
            x0 = np.zeros(dim)
            fx0 = func(x0)
            calls += 1
            update_best(x0, fx0)
            return best_x, fbest

        # --- Phase 2: Local Refinement via Powell ---
        remaining_budget = budget - calls
        if remaining_budget > 10 and time_ok():
            
            def powell_objective(x):
                nonlocal calls, best_x, fbest
                if not time_ok():
                    raise RuntimeError("Time limit exceeded")
                if calls >= budget:
                    raise RuntimeError("Budget exceeded")
                
                calls += 1
                fx = func(x)
                update_best(x, fx)
                return fx

            try:
                res = minimize(
                    powell_objective,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': remaining_budget, # Generous, cut by exception
                        'xtol': 1e-8,
                        'ftol': 1e-8
                    }
                )
            except RuntimeError:
                pass
            except Exception:
                pass

        return best_x, fbest
