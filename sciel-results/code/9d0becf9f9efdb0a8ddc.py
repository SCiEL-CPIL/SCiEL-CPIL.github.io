"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-Start Nelder-Mead optimizer using SciPy."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        stop_flag = [False]

        def evaluate(x):
            nonlocal calls_made
            if calls_made >= budget:
                stop_flag[0] = True
                return np.inf
            calls_made += 1
            return func(x)

        def callback(xk):
            if calls_made >= budget:
                stop_flag[0] = True
                return True
            return False

        # Phase 1: Initial Random Sampling
        # Use 10% of budget or at least 5 calls for initial exploration
        num_initial = max(5, min(20, budget // 10))
        
        initial_points = []
        for _ in range(num_initial):
            if calls_made >= budget:
                break
            x = rng.uniform(-5, 5, n)
            f = evaluate(x)
            initial_points.append((f, x))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            return np.zeros(n), np.inf

        initial_points.sort(key=lambda item: item[0])
        
        # Phase 2: Multi-Start Nelder-Mead
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f

        # Try Nelder-Mead from the top 3 initial points
        for i, (f_init, x_init) in enumerate(initial_points[:3]):
            if calls_made >= budget:
                break
            
            # Only run if this initial point is promising or if it's the first one
            if i > 0 and f_init > best_f + 10.0: # Heuristic: only try if reasonably close
                continue
                
            stop_flag[0] = False
            
            # Set max iterations based on remaining budget
            # Conservative estimate: 20 evals per iteration
            max_iter = max(10, remaining_budget // 20)
            
            # Create initial simplex
            simplex = [x_init.copy()]
            for d in range(n):
                x_pert = x_init.copy()
                # Step size: 1.0 is reasonable for range [-5, 5]
                x_pert[d] += 1.0
                x_pert = np.clip(x_pert, -5, 5)
                simplex.append(x_pert)
            
            try:
                res = minimize(
                    lambda x: evaluate(x), 
                    x_init, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'initial_simplex': np.array(simplex)
                    },
                    callback=callback
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

            if stop_flag[0]:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
