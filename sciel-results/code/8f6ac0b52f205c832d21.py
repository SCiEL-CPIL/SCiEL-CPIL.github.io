"""Optimizer for BBOB-style f12 (Bent cigar) using Nelder-Mead + (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Nelder-Mead for valley navigation + fine-tuning ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill

        # Evaluation counter
        calls = [0]

        def eval(x):
            calls[0] += 1
            return float(func(np.asarray(x, dtype=float)))

        # --- Phase 1: Global exploration to find a good starting basin ---
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Initial diverse sampling
        n_init = min(30, budget)
        for i in range(n_init):
            if calls[0] >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = eval(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = eval(best_x)

        # --- Phase 2: Nelder-Mead Local Optimization ---
        # Nelder-Mead is effective for the Bent Cigar function as it can form
        # a simplex aligned with the narrow valley, allowing rapid progress
        # along the valley floor compared to coordinate-wise methods.
        
        # Initialize simplex around best_x
        # Use a small initial step size to stay local
        init_step = 0.5 
        simplex = [best_x.copy()]
        for i in range(dim):
            pt = best_x.copy()
            pt[i] += init_step
            simplex.append(np.clip(pt, -5, 5))
        
        # Evaluate initial simplex
        simplex_vals = []
        for pt in simplex:
            if calls[0] >= budget or time.time() > deadline:
                break
            simplex_vals.append(eval(pt))
        
        if len(simplex_vals) < dim + 1:
            # Not enough budget for full simplex, fall back to ES immediately
            pass
        else:
            # Sort simplex by function value
            order = np.argsort(simplex_vals)
            simplex = [simplex[i] for i in order]
            simplex_vals = [simplex_vals[i] for i in order]
            
            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            # Main loop
            while calls[0] < budget - 50 and time.time() < deadline:
                # Ensure sorted
                order = np.argsort(simplex_vals)
                simplex = [simplex[i] for i in order]
                simplex_vals = [simplex_vals[i] for i in order]
                
                # Check for convergence
                if abs(simplex_vals[-1] - simplex_vals[0]) < 1e-8:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                worst_pt = simplex[-1]
                worst_val = simplex_vals[-1]
                
                # Reflection
                x_ref = centroid + alpha * (centroid - worst_pt)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = eval(x_ref)
                
                if f_ref < simplex_vals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = eval(x_exp)
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_vals[-1] = f_ref
                elif f_ref < simplex_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    simplex_vals[-1] = f_ref
                else:
                    # Contraction
                    # Outside contraction
                    x_con = centroid + rho * (worst_pt - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = eval(x_con)
                    
                    if f_con < worst_val:
                        simplex[-1] = x_con
                        simplex_vals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls[0] >= budget or time.time() > deadline:
                                break
                            simplex_vals[i] = eval(simplex[i])
                
                # Update global best
                current_min_idx = np.argmin(simplex_vals)
                if simplex_vals[current_min_idx] < best_f:
                    best_f = simplex_vals[current_min_idx]
                    best_x = simplex[current_min_idx].copy()

        # --- Phase 3: Fine-tuning with (1+1)-ES on remaining budget ---
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            sigma_es = 0.1
            x = best_x.copy()
            fx = best_f
            
            # Adaptive step size for ES
            for _ in range(budget - calls[0]):
                if calls[0] >= budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma_es * rng.standard_normal(dim), -5, 5)
                fy = eval(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma_es *= 1.2
                else:
                    sigma_es *= 0.9
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
