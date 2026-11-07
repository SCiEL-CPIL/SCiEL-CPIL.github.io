import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB f9 (rotated Rosenbrock).
    Uses a grid-based multi-start strategy:
    1. Evaluate a sparse 3^5 grid (243 points) to identify top basins.
    2. Use remaining budget for aggressive Nelder-Mead local optimizations from top grid points.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We stop at 300s to be safe.
        deadline = start_time + 300.0

        calls_made = 0
        best_x = None
        best_f = np.inf

        def wrapper(x):
            nonlocal calls_made, best_x, best_f
            # Time check
            if time.time() > deadline:
                return best_f
            # Budget check
            if calls_made >= budget:
                return best_f

            x = np.clip(x, -5.0, 5.0)
            calls_made += 1

            try:
                f_val = func(x)
            except Exception:
                return best_f

            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Grid-based global search
        # Create a 3^5 grid. Values: -4, 0, 4
        grid_vals = np.array([-4.0, 0.0, 4.0])
        grid_points = np.array(np.meshgrid(*[grid_vals] * dim)).T.reshape(-1, dim)
        
        # Shuffle grid points for potentially better early discovery
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(grid_points))
        grid_points = grid_points[perm]

        # Store grid results to identify top basins
        grid_results = []
        
        for i in range(len(grid_points)):
            if calls_made >= budget or time.time() > deadline:
                break
            x = grid_points[i]
            f_val = wrapper(x)
            grid_results.append((f_val, x))

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = wrapper(best_x)
            grid_results.append((best_f, best_x))

        # Phase 2: Local Polish using Nelder-Mead from top candidates
        # Sort grid results by fitness
        grid_results.sort(key=lambda item: item[0])
        
        # Select top 5 distinct points (they are distinct by construction of grid)
        top_k = min(5, len(grid_results))
        top_points = [grid_results[i][1] for i in range(top_k)]
        
        # Remaining budget
        # We have used ~243 calls. Remaining ~57.
        # We will split this among the top starts.
        # Nelder-Mead typically uses 1 + dim calls per iteration.
        # For 5D, that's 6 calls/iter.
        # With 57 calls, we can do ~9 iterations per start if we split evenly.
        # Or we can prioritize the best start.
        
        # Strategy: Give more budget to the best start.
        # Start 1: 50% of remaining
        # Start 2: 30% of remaining
        # Start 3: 10% of remaining
        # Start 4-5: 5% each
        
        remaining_budget = budget - calls_made
        if remaining_budget > 0 and time.time() < deadline - 2.0:
            # Define budget allocation ratios
            ratios = [0.4, 0.3, 0.15, 0.1, 0.05]
            
            for i in range(top_k):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                alloc = int(remaining_budget * ratios[i])
                if alloc < 10: # Minimum useful budget for NM
                    alloc = 10
                
                # Create a local wrapper that respects the specific allocation?
                # No, the global wrapper handles the total budget.
                # We just need to limit the maxiter of NM to consume approx 'alloc' calls.
                # NM calls func roughly (dim+1) times per iteration.
                # So maxiter ~ alloc / (dim+1)
                
                maxiter_nm = max(1, alloc // (dim + 1))
                
                try:
                    res = minimize(
                        wrapper,
                        top_points[i],
                        method='Nelder-Mead',
                        options={
                            'maxiter': maxiter_nm,
                            'xatol': 1e-8,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                except Exception:
                    pass

        return best_x, best_f
