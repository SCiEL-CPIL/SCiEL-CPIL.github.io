"""Model-and-optimize approach for BBOB-style Schwefel f20 on [-5,5]^5."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    """Hybrid Global-Local optimization using Differential Evolution."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_exceeded():
            return time.time() > deadline

        # We will split the budget:
        # ~80% for Global Search (Differential Evolution)
        # ~20% for Local Refinement (Nelder-Mead)
        
        # Differential Evolution parameters
        # popsize is a multiplier for dim. 
        # Standard is 15, but for small budget, smaller is better to allow more generations.
        # Let's try popsize=10. Population size = 10 * dim = 50.
        # If we have 400 evals, that's 8 generations.
        # Let's aim for 400 evals in DE.
        
        de_budget = int(budget * 0.8)
        local_budget = budget - de_budget
        
        popsize = 10
        # maxiter in DE is number of generations.
        # Each generation uses population_size evaluations.
        # population_size = popsize * dim
        pop_size = popsize * dim
        max_iter = max(1, de_budget // pop_size)
        
        # Define bounds
        bounds = [(-5.0, 5.0)] * dim
        
        # Wrapper for func to count evals and handle time
        eval_count = 0
        def wrapped_func(x):
            nonlocal eval_count
            if time_exceeded() or eval_count >= de_budget:
                return 1e10 # Penalize if over budget/time
            try:
                val = func(x)
                eval_count += 1
                return val
            except Exception:
                eval_count += 1
                return 1e10

        best_x = None
        best_f = np.inf
        
        try:
            # Run Differential Evolution
            res_de = differential_evolution(
                wrapped_func,
                bounds,
                maxiter=max_iter,
                popsize=popsize,
                tol=1e-6,
                seed=42,
                init='latinhypercube',
                update_interval=1
            )
            
            if res_de.success or res_de.fun < best_f:
                best_f = res_de.fun
                best_x = res_de.x.copy()
                
        except Exception:
            # Fallback if DE fails or times out
            pass

        # If DE didn't find anything or we have budget left, do local refinement
        if best_x is None:
            # Fallback to random or origin
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
                eval_count += 1
            except Exception:
                best_f = np.inf

        # Phase 2: Local Refinement
        if eval_count < budget and not time_exceeded():
            def wrapper_local(x_eval):
                nonlocal eval_count
                if time_exceeded() or eval_count >= budget:
                    return best_f + 1e10
                try:
                    val = func(x_eval)
                    eval_count += 1
                    return val
                except Exception:
                    eval_count += 1
                    return best_f + 1e10

            # Use Nelder-Mead for local polish
            # maxiter for NM is iterations, each uses ~dim+1 evals
            # We have local_budget left.
            # Let's be conservative.
            nm_max_iter = max(1, local_budget // (dim + 1))
            
            try:
                res_local = minimize(
                    wrapper_local,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': nm_max_iter,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                if res_local.fun < best_f:
                    best_f = res_local.fun
                    best_x = res_local.x.copy()
            except Exception:
                pass

        return best_x, best_f
