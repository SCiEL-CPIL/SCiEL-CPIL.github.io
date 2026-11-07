"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts for BBOB f9."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        best_x = None
        best_fx = np.inf
        calls_made = 0
        
        # Phase 1: Broad exploration with random starts
        # Phase 2: Local refinement with Nelder-Mead from best points
        
        # We'll use a pool of candidate starting points
        # and run Nelder-Mead on the most promising ones
        
        # Initial random sampling to find good basins
        n_initial = 30
        candidates = []
        
        for i in range(n_initial):
            if time.time() > deadline:
                break
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            calls_made += 1
            candidates.append((fx, x))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Take top few candidates for local refinement
        n_local = min(5, len(candidates))
        local_starts = [c[1] for c in candidates[:n_local]]
        
        # Run Nelder-Mead from each local start
        # Each NM run has a budget of remaining_calls / n_local
        remaining_calls = self.budget - calls_made
        nm_budget_per_run = max(10, remaining_calls // n_local)
        
        for start_x in local_starts:
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break
            
            # Run Nelder-Mead with a limited number of function evaluations
            # We need to track calls carefully
            
            nm_calls = [0]
            nm_budget = min(nm_budget_per_run, self.budget - calls_made)
            
            def nm_func(x):
                nonlocal calls_made, best_x, best_fx
                if nm_calls[0] >= nm_budget:
                    # Return a large value to stop NM
                    return 1e10
                nm_calls[0] += 1
                calls_made += 1
                fx = func(x)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                return fx
            
            try:
                result = minimize(
                    nm_func,
                    start_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': nm_budget,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                # Update best from result
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
            
            if time.time() > deadline:
                break
        
        # If we have budget left, do a few more random restarts
        # to see if we missed a better basin
        while calls_made < self.budget and time.time() < deadline:
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
                # Try a quick local refinement
                if time.time() < deadline - 5:
                    nm_calls = [0]
                    nm_budget = min(20, self.budget - calls_made)
                    
                    def nm_func2(x):
                        nonlocal calls_made, best_x, best_fx
                        if nm_calls[0] >= nm_budget:
                            return 1e10
                        nm_calls[0] += 1
                        calls_made += 1
                        fx = func(x)
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        return fx
                    
                    try:
                        result = minimize(
                            nm_func2,
                            x,
                            method='Nelder-Mead',
                            options={
                                'maxiter': nm_budget,
                                'xatol': 1e-6,
                                'fatol': 1e-8,
                                'adaptive': True
                            }
                        )
                        if result.fun < best_fx:
                            best_fx = result.fun
                            best_x = result.x.copy()
                    except Exception:
                        pass
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
