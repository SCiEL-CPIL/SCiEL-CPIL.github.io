"""
Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using multi-start Powell's method
with stratified Latin Hypercube initialization and L-BFGS-B polish.
"""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        # Helper to update best
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Dense stratified Latin Hypercube initialization
        # Use stratified sampling to cover the space well
        n_lh = 30
        # Create stratified samples: divide each dimension into n_lh strata
        # and sample one point from each stratum
        lh_points = []
        for i in range(n_lh):
            x = np.zeros(dim)
            for d in range(dim):
                # Stratified: sample uniformly in [i/n_lh, (i+1)/n_lh) mapped to [-5, 5]
                u = (i + np.random.uniform()) / n_lh
                x[d] = -5 + 10 * u
            lh_points.append(x)
        
        # Also add some random points for additional coverage
        n_extra = 10
        for _ in range(n_extra):
            x = np.random.uniform(-5, 5, size=dim)
            lh_points.append(x)
        
        # Evaluate all initial points
        for x in lh_points:
            if calls >= budget or time.time() > deadline:
                break
            f = func(x)
            calls += 1
            update_best(x, f)

        # Phase 2: Multi-start Powell's method with L-BFGS-B polish
        # Sort initial points by function value and use top ones as starting points
        # for local optimization
        
        # Collect evaluated points
        evaluated_points = []
        # Re-evaluate is not allowed, so we track from the LH phase
        # Actually we need to store them; let's restructure
        
        # Let's restart with a cleaner approach:
        # We'll do a two-phase approach:
        # 1. Global sampling (LH + random)
        # 2. Local polish on the best few points using Powell, then L-BFGS-B
        
        # Since we already evaluated, let's use the best points for local search
        # We need to track which points were evaluated. Let's redo this properly.
        
        # Actually, the above already called func. Let's just use the best_x
        # and do local polish from there, plus try a few other good starting points.
        
        # For a cleaner implementation, let's do:
        # 1. Evaluate a set of candidate points
        # 2. For the top K candidates, run Powell's method
        # 3. Polish with L-BFGS-B
        
        # Since we already spent calls on the LH points, let's now do local search
        # from the best point and a few others.
        
        # Get the best point
        if best_x is None:
            return best_x, best_f

        # Phase 2: Powell's method from best point
        # Powell's method is a derivative-free method that works well for
        # separable smooth functions like Rastrigin
        
        def powell_method(x0, max_calls, deadline, func, dim):
            """
            Powell's method implementation.
            Returns (best_x, best_f, calls_used)
            """
            nonlocal calls, best_x, best_f
            
            x = x0.copy()
            f = best_f  # We already have f for x0
            # Actually we need f for the starting point; let's use best_f
            
            # Direction vectors: start with standard basis
            directions = np.eye(dim)
            best_dir_f = f
            calls_used = 0
            
            max_iterations = 20
            
            for iteration in range(max_iterations):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Line search along each direction
                for i in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    dir_vec = directions[i]
                    # Golden section or parabolic line search
                    # Simple approach: try a few steps
                    
                    # First, find a good step size
                    step = 0.5
                    best_local_f = f
                    best_local_x = x.copy()
                    
                    # Try positive and negative directions
                    for sign in [1.0, -1.0]:
                        for _ in range(5):  # A few step sizes
                            if calls >= budget or time.time() > deadline:
                                break
                            x_trial = x + sign * step * dir_vec
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = func(x_trial)
                            calls += 1
                            calls_used += 1
                            if f_trial < best_local_f:
                                best_local_f = f_trial
                                best_local_x = x_trial.copy()
                            step *= 0.7  # Shrink step
                            if step < 1e-4:
                                break
                    
                    if best_local_f < f:
                        x = best_local_x
                        f = best_local_f
                        update_best(x, f)
                        best_dir_f = f
                        # Update direction to be the one that led to improvement
                        # (simplified: keep as is for now)
                
                # Check if we made progress
                if f >= best_dir_f - 1e-10:
                    # No significant progress, try to reduce step or break
                    # Simple stagnation check
                    if iteration > 5:
                        break
                    continue
                
                best_dir_f = f
            
            return x, f, calls_used

        # Run Powell from the best point
        x_powell, f_powell, _ = powell_method(best_x, budget, deadline, func, dim)
        
        # Phase 3: L-BFGS-B polish
        if calls < budget and time.time() < deadline and best_x is not None:
            # Use L-BFGS-B for final polish
            # This is a gradient-based method that works well near minima
            try:
                result = minimize(
                    lambda x: func(np.clip(x, -5, 5)),
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfun': min(100, budget - calls)}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # Phase 4: If we still have budget, try Powell from a few other good points
        # This helps with the bi-modal structure
        if calls < budget and time.time() < deadline:
            # Try a few random restarts with local polish
            n_restarts = 3
            for _ in range(n_restarts):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Random restart near current best
                x_start = best_x + np.random.normal(0, 1.0, size=dim)
                x_start = np.clip(x_start, -5, 5)
                
                f_start = func(x_start)
                calls += 1
                update_best(x_start, f_start)
                
                # Quick Powell from here
                x_p, f_p, _ = powell_method(x_start, budget, deadline, func, dim)
                
                # L-BFGS-B polish
                if calls < budget and time.time() < deadline:
                    try:
                        result = minimize(
                            lambda x: func(np.clip(x, -5, 5)),
                            best_x,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * dim,
                            options={'maxiter': 30, 'maxfun': min(50, budget - calls)}
                        )
                        if result.fun < best_f:
                            best_f = result.fun
                            best_x = result.x.copy()
                    except Exception:
                        pass
        
        return best_x, best_f
