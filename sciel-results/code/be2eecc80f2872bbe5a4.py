"""Basin Hopping optimizer for BBOB f20 (Schwefel) with Nelder-Mead local refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """
    Basin Hopping optimizer with Nelder-Mead local refinement.
    Designed for BBOB f20 (Schwefel) on [-5,5]^5.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def try_func(x):
            nonlocal calls_made
            if calls_made >= total_budget:
                return None
            if time.time() > deadline:
                return None
            fx = func(x)
            calls_made += 1
            return fx

        # Generate diverse starting points
        # Include origin, known Schwefel min (approx), and random points
        starts = []
        # Known Schwefel global min is at x_i = 4.209687...
        # The problem states minimizer is in [-4, 4]^5, but hidden shift/rotation applies.
        # We include points near the standard min location and its negatives as heuristics.
        schwefel_min_std = np.array([4.20968746, 4.20968746, 4.20968746, 4.20968746, 4.20968746])
        # Clip to box as per problem statement
        starts.append(np.clip(schwefel_min_std, -5, 5))
        starts.append(np.clip(-schwefel_min_std, -5, 5))
        starts.append(np.zeros(dim))
        
        # Add structured points
        starts.append(np.ones(dim) * 2.0)
        starts.append(np.ones(dim) * -2.0)
        
        # Add random points
        rng_global = np.random.default_rng(42)
        for _ in range(5):
            starts.append(rng_global.uniform(-4.5, 4.5, dim))

        # Basin Hopping Parameters
        # We will perform a series of jumps.
        # Each jump: 
        # 1. Pick a new center (either a start point or a perturbation of the current best)
        # 2. Run local Nelder-Mead from that center with a limited budget
        # 3. If the local min is better, update global best
        
        # Budget allocation:
        # Reserve some calls for the final polish if needed, but mostly use for basin hopping.
        # Let's say each local NM run gets up to 50 calls.
        # With 500 total budget, we can afford ~10 local runs.
        local_budget_per_run = 50
        max_local_runs = total_budget // local_budget_per_run
        
        # We have 10 starts. We will try each one with a local run.
        # If we have budget left, we will do more random jumps.
        
        current_x = None
        current_f = float('inf')

        for i, start_pt in enumerate(starts):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break
            
            # Initialize local run
            x0 = start_pt.copy()
            
            # Wrap func for local minimization with budget tracking
            local_calls = [0]
            def local_func(x):
                nonlocal calls_made
                if local_calls[0] >= local_budget_per_run:
                    return float('inf')
                if calls_made >= total_budget:
                    return float('inf')
                if time.time() > deadline:
                    return float('inf')
                fx_val = func(x)
                local_calls[0] += 1
                calls_made += 1
                return fx_val

            try:
                res = minimize(
                    local_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': local_budget_per_run,
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                
                if res.fun < float('inf') and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    current_x = res.x.copy()
                    current_f = res.fun
                    
            except Exception:
                pass

        # If we have budget left, do random jumps from the best point found
        if best_x is not None and calls_made < total_budget:
            rng_jump = np.random.default_rng(12345)
            while calls_made < total_budget and time.time() < deadline:
                # Jump size: adaptive, start large, shrink if no improvement
                jump_sigma = 1.0
                x_jump = best_x + jump_sigma * rng_jump.standard_normal(dim)
                x_jump = np.clip(x_jump, -5, 5)
                
                # Run a smaller local refinement from the jump point
                local_budget_small = 20
                local_calls_small = [0]
                def local_func_small(x):
                    nonlocal calls_made
                    if local_calls_small[0] >= local_budget_small:
                        return float('inf')
                    if calls_made >= total_budget:
                        return float('inf')
                    if time.time() > deadline:
                        return float('inf')
                    fx_val = func(x)
                    local_calls_small[0] += 1
                    calls_made += 1
                    return fx_val

                try:
                    res = minimize(
                        local_func_small,
                        x_jump,
                        method='Nelder-Mead',
                        options={
                            'maxiter': local_budget_small,
                            'xatol': 1e-3,
                            'fatol': 1e-3,
                            'adaptive': True
                        }
                    )
                    
                    if res.fun < float('inf') and res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                        # If we found a better point, we can continue jumping from here
                        # Otherwise, we might want to try a larger jump or different direction
                    else:
                        # If no improvement, we can try to shrink the jump size slightly
                        # or just continue. For simplicity, we just continue.
                        pass
                        
                except Exception:
                    pass
                
                # Safety check to avoid infinite loop if budget is tight
                if local_calls_small[0] == 0:
                    break

        # Final Polish: If we have a few calls left, do a very tight NM on the best point
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            remaining = total_budget - calls_made
            if remaining > 10:
                local_calls_final = [0]
                def local_func_final(x):
                    nonlocal calls_made
                    if local_calls_final[0] >= remaining:
                        return float('inf')
                    if calls_made >= total_budget:
                        return float('inf')
                    if time.time() > deadline:
                        return float('inf')
                    fx_val = func(x)
                    local_calls_final[0] += 1
                    calls_made += 1
                    return fx_val

                try:
                    res = minimize(
                        local_func_final,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': remaining,
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    
                    if res.fun < float('inf') and res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Fallback: ensure at least one call was made
        if best_x is None:
            x = np.zeros(dim)
            best_f = func(x)
            best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
