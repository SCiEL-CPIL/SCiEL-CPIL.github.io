import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        # State variables
        best_x = None
        best_fx = np.inf
        calls_made = 0
        all_xs = []
        all_fxs = []

        # Use a fixed random seed for reproducibility
        rng = np.random.default_rng(0)

        def safe_func(x):
            nonlocal calls_made, best_x, best_fx
            # Clip to box
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            all_xs.append(x.copy())
            all_fxs.append(fx)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Surrogate-Assisted Global Search
        # Use first 40% of budget for initial sampling and surrogate-based acquisition
        surrogate_budget = int(budget * 0.4)
        
        # Initialize with a few random points to build initial surrogate
        for i in range(min(5, surrogate_budget)):
            x = rng.uniform(-5, 5, dim)
            safe_func(x)
        
        # Simple surrogate: Kriging/GP-like using distance-weighted average
        # For simplicity and speed, we'll use a lightweight approach:
        # Fit a simple radial basis function (RBF) interpolant or use k-NN regression
        
        # Since we can't easily use full GP in tight loop, we'll use a simpler heuristic:
        # Expected Improvement based on local variance estimation
        
        while calls_made < surrogate_budget:
            if time.time() > deadline:
                break
            
            if len(all_xs) < 3:
                # Not enough data, just sample randomly
                x = rng.uniform(-5, 5, dim)
            else:
                # Compute acquisition function: balance exploration (uncertainty) and exploitation (low f)
                # Use k-NN to estimate local mean and std
                xs = np.array(all_xs)
                fxs = np.array(all_fxs)
                
                # Sample candidate points
                n_candidates = 20
                candidates = rng.uniform(-5, 5, (n_candidates, dim))
                
                # For each candidate, compute distance to nearest k points
                k = min(5, len(xs))
                scores = np.zeros(n_candidates)
                
                for i, cand in enumerate(candidates):
                    dists = np.linalg.norm(xs - cand, axis=1)
                    nearest_idx = np.argsort(dists)[:k]
                    nearest_fxs = fxs[nearest_idx]
                    nearest_dists = dists[nearest_idx]
                    
                    # Weighted average for mean
                    weights = 1.0 / (nearest_dists + 1e-8)
                    weights /= np.sum(weights)
                    local_mean = np.sum(weights * nearest_fxs)
                    local_std = np.std(nearest_fxs)
                    
                    # Expected Improvement
                    improvement = best_fx - local_mean
                    if improvement > 0:
                        scores[i] = improvement * 0.8 + local_std * 0.2
                    else:
                        scores[i] = local_std * 0.5
                
                x = candidates[np.argmax(scores)]
            
            x = np.clip(x, -5, 5)
            safe_func(x)

        # Phase 2: Local Refinement with Multi-Start Nelder-Mead
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and best_x is not None and time.time() < deadline:
            # Use remaining budget for local search with multiple starts
            n_starts = min(5, remaining_budget // 10)
            budget_per_start = remaining_budget // n_starts
            
            for start_idx in range(n_starts):
                if calls_made >= budget - 5 or time.time() > deadline:
                    break
                
                # Start from best point with perturbation
                if start_idx == 0:
                    x0 = best_x.copy()
                else:
                    x0 = best_x + rng.uniform(-0.5, 0.5, dim)
                    x0 = np.clip(x0, -5, 5)
                
                local_calls = 0
                def local_func(x):
                    nonlocal local_calls
                    if local_calls >= budget_per_start:
                        return best_fx
                    local_calls += 1
                    return safe_func(x)
                
                try:
                    res = minimize(
                        local_func, 
                        x0, 
                        method='Nelder-Mead', 
                        options={'maxiter': budget_per_start, 'xtol': 1e-8, 'ftol': 1e-8}
                    )
                    if res.fun < best_fx:
                        best_fx = res.fun
                        best_x = res.x
                except Exception:
                    pass

        return best_x, best_fx
