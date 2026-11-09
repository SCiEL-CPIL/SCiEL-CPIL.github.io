"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart local search using Nelder-Mead for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Track all evaluated points to calculate hypervolume later if needed, 
        # but the evaluator records all calls to problem.
        # We just need to ensure we don't exceed budget.
        
        # Strategy: 
        # 1. Generate a set of diverse starting points (grid + random)
        # 2. For each starting point, run a limited local search (Nelder-Mead)
        #    using a weighted sum scalarization to find a Pareto point.
        #    We vary the weights to cover the Pareto front.
        
        # Budget allocation:
        # Let's say we have B total evaluations.
        # We'll use N restarts. Each restart uses roughly B/N evaluations.
        # Nelder-Mead with dim=4 is fast. 
        # Let's try N=10 restarts.
        # Each restart: 1 initial eval + (B/N - 1) local steps.
        
        n_restarts = 10
        evals_per_restart = self.budget // n_restarts
        
        # Generate diverse starting points
        # Mix of grid and random
        starts = []
        # 5 grid points (corners and center-ish)
        grid_vals = [0.1, 0.5, 0.9]
        for x1 in grid_vals:
            for x2 in grid_vals:
                if len(starts) < 5:
                    starts.append(np.array([x1, x2, 0.5, 0.5]))
        
        # Fill rest with random
        rng = np.random.default_rng(0)
        while len(starts) < n_restarts:
            starts.append(lo + (hi - lo) * rng.random(dim))
        
        # Define different weight vectors for scalarization to explore different parts of PF
        # We want to cover the front.
        # Simple linear weights.
        weights = []
        for i in range(n_restarts):
            # Uniform distribution over simplex
            w = rng.dirichlet(np.ones(n_obj))
            weights.append(w)
        
        used_evals = 0
        all_points = []
        
        for i in range(n_restarts):
            if used_evals >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x0 = starts[i]
            w = weights[i]
            
            # Scalarized objective: sum(w_j * f_j(x))
            # But we need to track individual evaluations? 
            # The evaluator records every call to problem.
            # The local optimizer will call problem multiple times.
            # We need to count these calls against the budget.
            
            # scipy minimize with method='Nelder-Mead'
            # options: maxiter, maxfev
            # maxfev controls function evaluations.
            
            # We need a wrapper that counts evaluations and stops if budget exceeded.
            # However, scipy's minimize doesn't easily interrupt mid-iteration based on external budget 
            # unless we raise an exception or use a callback that raises.
            
            # Let's use a callback to check time/budget.
            # But maxfev is the main control.
            # We have `evals_per_restart` available for this restart.
            # Note: x0 evaluation is done by minimize.
            
            def obj(x):
                nonlocal used_evals
                if used_evals >= self.budget:
                    raise ValueError("Budget exceeded")
                fx = problem(x)
                used_evals += 1
                return np.dot(w, fx)
            
            try:
                res = minimize(obj, x0, method='Nelder-Mead', 
                               options={'maxiter': evals_per_restart, 'maxfev': evals_per_restart, 'xatol': 1e-4, 'fatol': 1e-4})
                # The result res.x is the best point found by local search.
                # We should record this point explicitly? 
                # The evaluator records ALL calls to problem. 
                # So the intermediate points from Nelder-Mead are already recorded.
                # We don't need to do anything else for recording.
            except ValueError:
                break
            except Exception:
                # If minimize fails for other reasons, continue to next restart
                continue

        return None
# EVOLVE-BLOCK-END
