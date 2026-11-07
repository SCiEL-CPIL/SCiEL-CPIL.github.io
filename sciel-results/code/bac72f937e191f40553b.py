"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using Tchebycheff scalarization with
    sequential simplex-based local optimization from diverse starting points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Generate diverse Tchebycheff weight vectors
        # For 2 objectives, weights are (w, 1-w) for w in some set
        # We'll use a mix of uniform and some extreme weights
        n_weights = 12
        # Include extremes and intermediate values
        weights = np.linspace(0.05, 0.95, n_weights)
        
        # Reserve some budget for initial random exploration
        initial_budget = 20
        remaining = budget - initial_budget
        per_weight = remaining // n_weights
        extra = remaining % n_weights
        
        # Store all evaluated points
        X_all = []
        F_all = []
        used = 0
        
        def eval_point(x):
            nonlocal used
            if used >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            X_all.append(x.copy())
            F_all.append(np.array(f))
            return f
        
        # Phase 1: Initial random exploration
        for i in range(initial_budget):
            if time.time() > deadline - 10:
                break
            x = lo + span * rng.random(dim)
            eval_point(x)
        
        # Phase 2: Tchebycheff scalarization with local optimization
        # For each weight vector, do a few local optimizations from random starts
        
        def tchebycheff_objective(x, w):
            f = problem(x)
            # Find nadir point (worst value for each objective) from current points
            # Use a simple approach: max of all seen values
            if F_all:
                F_arr = np.array(F_all)
                nadir = np.max(F_arr, axis=0)
            else:
                nadir = np.array([1.0] * n_obj)
            
            # Tchebycheff scalarization: max_i w_i * |f_i - nadir_i|
            # Since we're minimizing, we want to minimize max_i w_i * (f_i / nadir_i)
            # But nadir might be 0, so use a different normalization
            # Actually, standard Tchebycheff: min_x max_i w_i * |f_i(x) - z_i^utopia|
            # But we don't know utopia. Let's use a simpler approach.
            # Alternative: use weighted sum with adaptive weights based on progress
            # Or use achievement scalarizing function
            
            # Let's use a simpler weighted sum approach with adaptive scaling
            # Scale each objective by its current range
            if len(F_all) > 1:
                F_arr = np.array(F_all)
                obj_min = np.min(F_arr, axis=0)
                obj_max = np.max(F_arr, axis=0)
                obj_range = obj_max - obj_min
                obj_range[obj_range < 1e-10] = 1.0
                f_scaled = (f - obj_min) / obj_range
            else:
                f_scaled = f
            
            return np.max(w * f_scaled)
        
        # Better approach: use weighted sum with normalization based on observed range
        # and do Nelder-Mead optimization on the scalarized objective
        
        for wi_idx in range(n_weights):
            w = np.array([weights[wi_idx], 1.0 - weights[wi_idx]])
            n_starts = per_weight + (1 if wi_idx < extra else 0)
            
            for start_idx in range(n_starts):
                if time.time() > deadline - 5 or used >= budget:
                    break
                
                # Random starting point
                x0 = lo + span * rng.random(dim)
                f0 = eval_point(x0)
                if f0 is None:
                    break
                
                # Do a few iterations of local optimization using a simple
                # coordinate descent or Nelder-Mead-like approach
                # Since we can't use scipy.optimize directly (it would count
                # many evaluations), we'll implement a simple local search
                
                best_x = x0.copy()
                best_f = f0.copy()
                
                # Simple local search: try small perturbations in each direction
                n_local_iters = min(8, budget - used)
                step_sizes = [0.1, 0.05, 0.02, 0.01]
                
                for it in range(n_local_iters):
                    if used >= budget or time.time() > deadline - 2:
                        break
                    
                    improved = False
                    for d in range(dim):
                        for direction in [-1, 1]:
                            if used >= budget:
                                break
                            step = step_sizes[min(it // 2, len(step_sizes) - 1)] * span[d]
                            x_try = best_x.copy()
                            x_try[d] += direction * step
                            x_try = np.clip(x_try, lo, hi)
                            
                            f_try = problem(x_try)
                            used += 1
                            X_all.append(x_try.copy())
                            F_all.append(np.array(f_try))
                            
                            # Check if this improves the Tchebycheff scalarization
                            # Compute scalarized value for both
                            if len(F_all) >= 2:
                                F_arr = np.array(F_all)
                                obj_min = np.min(F_arr, axis=0)
                                obj_max = np.max(F_arr, axis=0)
                                obj_range = obj_max - obj_min
                                obj_range[obj_range < 1e-10] = 1.0
                                f_try_scaled = (f_try - obj_min) / obj_range
                                f_best_scaled = (best_f - obj_min) / obj_range
                                scalar_try = np.max(w * f_try_scaled)
                                scalar_best = np.max(w * f_best_scaled)
                            else:
                                scalar_try = np.max(w * f_try)
                                scalar_best = np.max(w * best_f)
                            
                            if scalar_try < scalar_best - 1e-10:
                                best_x = x_try
                                best_f = f_try
                                improved = True
                    
                    if not improved:
                        break
        
        # Convert to arrays
        if X_all:
            X_result = np.array(X_all)
            F_result = np.array(F_all)
        else:
            X_result = np.zeros((0, dim))
            F_result = np.zeros((0, n_obj))
        
        return X_result
# EVOLVE-BLOCK-END
