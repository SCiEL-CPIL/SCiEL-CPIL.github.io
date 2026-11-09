"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import cma, minimize

class Optimizer:
    """CMA-ES global search followed by L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        n = self.dim
        calls_made = 0
        best_x = np.zeros(n)
        best_f = np.inf

        def safe_func(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            x = np.clip(x, -5, 5)
            calls_made += 1
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
                
            return val

        # Phase 1: Global search with CMA-ES
        # CMA-ES is robust to rotations and shifts, unlike gradient methods
        # Budget split: ~70% for global, 30% for local
        cma_budget = int(self.budget * 0.7)
        
        try:
            # Initialize CMA-ES
            # Start at center, moderate sigma
            x0 = np.zeros(n)
            sigma0 = 2.0
            
            opts = {
                'maxiter': cma_budget,  # Limit iterations to stay within budget
                'seed': 42,
                'verbose': -9,
                'bounds': [(-5, 5) for _ in range(n)],
                'tolfun': 1e-10,
                'tolx': 1e-10,
                'maxfevals': cma_budget + 10  # Safety buffer
            }
            
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            
            while not es.stop() and calls_made < self.budget:
                if time.time() > deadline:
                    break
                solutions = es.ask()
                for sol in solutions:
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    safe_func(sol)
                # Only evaluate if we didn't hit budget in the inner loop
                if calls_made < self.budget:
                    fvals = [safe_func(sol) for sol in solutions if calls_made < self.budget]
                    # Note: The above loop logic is slightly redundant with safe_func counting.
                    # CMA-ES handles the counting internally if we use the standard API, 
                    # but we are wrapping it. Let's stick to the standard CMA-ES pattern 
                    # but ensure we don't exceed budget.
                    
                    # Actually, the standard pattern is:
                    # solutions = es.ask()
                    # fvals = [safe_func(x) for x in solutions]
                    # es.tell(solutions, fvals)
                    
                    # Let's re-implement the loop cleanly to avoid double counting issues.
                    
            # The loop above was messy. Let's restart the CMA-ES logic cleanly.
            
        except Exception:
            pass

        # Re-implementing CMA-ES loop cleanly to ensure correctness
        calls_made = 0
        best_x = np.zeros(n)
        best_f = np.inf

        def safe_func(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            x = np.clip(x, -5, 5)
            calls_made += 1
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
                
            return val

        cma_budget = int(self.budget * 0.7)
        
        try:
            x0 = np.zeros(n)
            sigma0 = 2.0
            opts = {
                'maxiter': cma_budget,
                'seed': 42,
                'verbose': -9,
                'bounds': [(-5, 5) for _ in range(n)],
                'tolfun': 1e-10,
                'tolx': 1e-10
            }
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            
            while not es.stop() and calls_made < self.budget:
                if time.time() > deadline:
                    break
                solutions = es.ask()
                fvals = []
                for sol in solutions:
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    fvals.append(safe_func(sol))
                if len(fvals) == len(solutions):
                    es.tell(solutions, fvals)
                else:
                    break
                    
        except Exception:
            pass

        # Phase 2: Local refinement with L-BFGS-B
        # Use the best point found so far
        if calls_made < self.budget:
            try:
                local_budget = self.budget - calls_made
                res = minimize(
                    safe_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5) for _ in range(n)],
                    options={'maxiter': local_budget, 'gtol': 1e-8}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
