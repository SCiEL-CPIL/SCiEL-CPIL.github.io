import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        eval_history = [] # List of (f, x)

        def safe_call(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            eval_history.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Exploration using Sobol Sequence
        # Sobol sequences are low-discrepancy and good for global search
        n_explore = int(budget * 0.4)
        
        # Generate Sobol points
        # qmc.Sobol requires a power of 2 for sample size for best quality, but works otherwise
        # We will generate n_explore points
        try:
            sampler = qmc.Sobol(d=dim, scramble=True)
            sobol_samples = sampler.random(n_explore)
        except Exception:
            # Fallback to Halton if Sobol fails
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f = f / base
                    r = r + f * (index % base)
                    index = index // base
                return r
            primes = [2, 3, 5, 7, 11]
            sobol_samples = np.array([[halton(i, primes[d]) for d in range(dim)] for i in range(1, n_explore + 1)])
            
        # Map [0, 1] to [-5, 5]
        sobol_samples = sobol_samples * 10.0 - 5.0
        
        # Ensure we sample the origin
        origin = np.zeros(dim)
        if not np.any(np.isclose(sobol_samples, origin, atol=1e-6).all(axis=1)):
             # Prepend origin
             sobol_samples = np.vstack([origin, sobol_samples])
             n_explore += 1

        # Evaluate all candidates
        for x in sobol_samples:
            if calls_made >= budget or time.time() > deadline - 10.0:
                break
            safe_call(x)

        # Phase 2: Local Refinement using Nelder-Mead on the best candidates
        if eval_history:
            # Sort by function value
            eval_history.sort(key=lambda item: item[0])
            
            # Select top K points
            top_k = 5
            # Ensure we don't pick more points than we have
            start_points = [x for f, x in eval_history[:top_k]]
            
            # Distribute remaining budget among start points
            remaining_budget = budget - calls_made
            if remaining_budget > 0 and start_points:
                # Give more budget to the best point, less to others
                # Simple heuristic: divide evenly, but cap iterations
                budget_per_point = remaining_budget // len(start_points)
                
                for i, x0 in enumerate(start_points):
                    if calls_made >= budget or time.time() > deadline - 5.0:
                        break
                    
                    # Reduce budget for later points if time is tight
                    current_local_budget = budget_per_point
                    if i > 0:
                         current_local_budget = max(10, int(budget_per_point * 0.8))

                    local_calls = 0
                    local_budget = current_local_budget
                    
                    def nm_func(x):
                        nonlocal local_calls
                        if local_calls >= local_budget or calls_made >= budget:
                            return np.inf
                        local_calls += 1
                        return safe_call(x)
                    
                    try:
                        res = minimize(
                            nm_func,
                            x0=x0,
                            method='Nelder-Mead',
                            options={
                                'maxiter': 200,
                                'xatol': 1e-8,
                                'fatol': 1e-8,
                                'adaptive': True
                            }
                        )
                    except Exception:
                        pass

        return best_x, best_f
