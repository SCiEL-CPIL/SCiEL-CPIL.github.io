import numpy as np
from scipy.spatial import KDTree
from scipy.stats import qmc
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        limit = 330 - 30  # 300s margin + safety
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        # Storage for surrogate model
        X_obs = []
        Y_obs = []

        def wrapper(x):
            nonlocal calls_made, best_x, best_f, X_obs, Y_obs
            if time.time() - start_time > limit:
                return best_f
            if calls_made >= self.budget:
                return best_f
            
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                return best_f
            
            if np.isnan(f_val) or np.isinf(f_val):
                f_val = best_f if best_f != float('inf') else 1e10
            
            X_obs.append(x.copy())
            Y_obs.append(f_val)
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            
            return f_val

        # Phase 1: Surrogate-Assisted Search
        # Use ~80% of budget for surrogate, 20% for local polish
        surrogate_budget = int(self.budget * 0.8)
        
        # 1. Initialization with Latin Hypercube Sampling
        n_init = 30
        if n_init > surrogate_budget:
            n_init = surrogate_budget
            
        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
        init_samples = sampler.random(n=n_init) * 10 - 5 # Scale to [-5, 5]
        
        for x in init_samples:
            if time.time() - start_time > limit:
                break
            wrapper(x)
            
        # 2. Surrogate Loop
        # We will generate a batch of candidates each step to pick the best UCB
        n_candidates = 100
        k_neighbors = 10
        beta = 1.5 # Exploration weight
        
        while calls_made < surrogate_budget:
            if time.time() - start_time > limit:
                break
                
            if len(X_obs) < 2:
                # If too few points, just do random search
                x_new = np.random.uniform(-5, 5, self.dim)
                wrapper(x_new)
                continue
                
            X_obs_arr = np.array(X_obs)
            Y_obs_arr = np.array(Y_obs)
            
            # Build KDTree
            try:
                tree = KDTree(X_obs_arr)
            except Exception:
                break
                
            # Generate candidates
            # Mix of random and local perturbations around best point
            candidates = []
            # 50% global random
            n_global = n_candidates // 2
            sampler_cand = qmc.LatinHypercube(d=self.dim, seed=42 + calls_made)
            global_cands = sampler_cand.random(n=n_global) * 10 - 5
            candidates.extend(global_cands)
            
            # 50% local perturbations around best_x
            if best_x is not None:
                n_local = n_candidates - n_global
                # Scale perturbation based on current best value? 
                # For now, fixed scale 1.0
                local_cands = best_x + np.random.normal(0, 1.0, (n_local, self.dim))
                # Clip to box
                local_cands = np.clip(local_cands, -5, 5)
                candidates.extend(local_cands)
            else:
                # If no best_x yet, just more global
                sampler_cand2 = qmc.LatinHypercube(d=self.dim, seed=43 + calls_made)
                more_global = sampler_cand2.random(n=n_candidates) * 10 - 5
                candidates.extend(more_global)
                
            candidates = np.array(candidates)
            
            # Evaluate UCB for candidates
            # Distances to k nearest neighbors
            distances, indices = tree.query(candidates, k=k_neighbors)
            
            # Get Y values for neighbors
            # indices shape: (n_candidates, k_neighbors)
            neighbor_Y = Y_obs_arr[indices]
            
            # Mean and Std
            means = np.mean(neighbor_Y, axis=1)
            stds = np.std(neighbor_Y, axis=1)
            
            # UCB
            ucb_scores = means + beta * stds
            
            # Pick best candidate (minimize UCB? No, UCB is for maximization of uncertainty/improvement)
            # We want to minimize the function. 
            # Standard EI/UCB for minimization:
            # We want low mean (exploitation) and high uncertainty (exploration).
            # So we want to MINIMIZE: mean - beta * std? 
            # Or MAXIMIZE: -mean + beta * std?
            # Let's use: Score = -mean + beta * std. We pick the candidate with MAX Score.
            # This balances finding low values (low mean) with exploring uncertain areas (high std).
            scores = -means + beta * stds
            
            best_idx = np.argmax(scores)
            x_new = candidates[best_idx]
            
            wrapper(x_new)
            
            # Adaptive Beta: reduce exploration as we converge?
            # Simple heuristic: reduce beta slightly over time
            # beta = max(0.5, beta * 0.99) 
            # Let's keep beta fixed for simplicity in this version, 
            # or adjust based on improvement.
            
        # Phase 2: Local Polish
        # Use remaining budget
        if best_x is not None and calls_made < self.budget:
            try:
                # Powell is robust for Rosenbrock
                res_powell = minimize(
                    wrapper, 
                    best_x, 
                    method='Powell', 
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if res_powell.fun < best_f:
                    best_f = res_powell.fun
                    best_x = res_powell.x.copy()
            except Exception:
                pass
        
        # Final safety check
        if time.time() - start_time > limit:
            pass # Just return what we have

        return best_x, best_f
