import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 280  # Safety margin
            
            dim = self.dim
            budget = self.budget
            
            # Multi-Start Nelder-Mead
            # 5 independent simplexes from diverse, space-filling initial points
            
            # Generate 5 starting points using Latin Hypercube Sampling (LHS)
            n_starts = 5
            # Simple LHS: divide each dimension into n_starts bins, pick one point per bin
            # For simplicity and robustness, we can use a fixed set of diverse points or random LHS
            # Let's use a deterministic set of points spread across the box to ensure coverage
            # Points: Center, Corners (scaled), and some mid-points
            # Since dim=5, 5 points is a good start.
            
            # A better approach for LHS in 5D with 5 points:
            # Sort indices for each dimension to create a permutation
            # But simple random with seed is also fine. Let's use a fixed diverse set.
            
            # Diverse points in [-5, 5]^5
            # 1. Origin (0,0,0,0,0)
            # 2. (4, 4, 4, 4, 4)
            # 3. (-4, -4, -4, -4, -4)
            # 4. (4, -4, 4, -4, 4)
            # 5. (-4, 4, -4, 4, -4)
            
            start_points = [
                np.zeros(dim),
                np.array([4.0]*dim),
                np.array([-4.0]*dim),
                np.array([4.0, -4.0, 4.0, -4.0, 4.0]),
                np.array([-4.0, 4.0, -4.0, 4.0, -4.0])
            ]
            
            # If dim != 5, adjust points
            if dim != 5:
                start_points = []
                for i in range(n_starts):
                    # Random points within box
                    p = np.random.uniform(-4.0, 4.0, dim)
                    start_points.append(p)
            else:
                # Ensure points are within [-5, 5]
                start_points = [np.clip(p, -5.0, 5.0) for p in start_points]

            best_x = start_points[0].copy()
            best_f = func(best_x)
            evals_used = 1
            
            # Budget per start
            # We have 500 total. 1 used. 499 left.
            # 5 starts -> ~99 evals per start.
            # Nelder-Mead in 5D needs simplex size 6.
            # Each iteration uses 1 eval.
            # 99 evals is plenty for NM to converge locally.
            
            evals_per_start = (budget - evals_used) // n_starts
            if evals_per_start < 10:
                evals_per_start = 10
                n_starts = (budget - evals_used) // evals_per_start
                if n_starts < 1:
                    n_starts = 1
                    evals_per_start = budget - evals_used

            # Nelder-Mead Parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            # Simplex size (initial step)
            # Since the function is scaled to [-5,5], a step of 1.0 or 0.5 is reasonable.
            # For "Different powers", the landscape can be steep.
            # Let's use a small initial simplex to avoid overshooting, but not too small.
            delta = 0.5
            
            for s_idx in range(n_starts):
                if time.time() > deadline:
                    break
                if evals_used >= budget:
                    break
                
                x0 = start_points[s_idx].copy()
                
                # Initialize simplex
                # x0 is the first vertex
                simplex = [x0.copy()]
                f_vals = [func(x0)]
                evals_used += 1
                
                # Create other vertices by adding delta to each coordinate
                for i in range(dim):
                    x_new = x0.copy()
                    x_new[i] += delta
                    simplex.append(x_new)
                    f_new = func(x_new)
                    f_vals.append(f_new)
                    evals_used += 1
                    if evals_used >= budget:
                        break
                
                if evals_used >= budget:
                    break
                
                # Track local best
                local_best_x = simplex[np.argmin(f_vals)].copy()
                local_best_f = min(f_vals)
                
                # Main Nelder-Mead Loop
                max_iter = evals_per_start - (len(simplex)) # approx
                # Actually, just loop until budget or time
                while evals_used < budget and time.time() < deadline:
                    # Sort simplex by function value
                    # indices = np.argsort(f_vals)
                    # Since dim is small, we can sort lists
                    pairs = list(zip(f_vals, simplex))
                    pairs.sort(key=lambda x: x[0])
                    f_vals = [p[0] for p in pairs]
                    simplex = [p[1] for p in pairs]
                    
                    # Check for convergence
                    if len(f_vals) > 1:
                        if abs(f_vals[-1] - f_vals[0]) < 1e-10:
                            break
                        # Also check if simplex is very small
                        if np.linalg.norm(simplex[-1] - simplex[0]) < 1e-8:
                            break
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5.0, 5.0)
                    f_r = func(x_r)
                    evals_used += 1
                    
                    if f_r < f_vals[0]:
                        # Expansion
                        x_e = centroid + gamma * (centroid - simplex[-1])
                        x_e = np.clip(x_e, -5.0, 5.0)
                        f_e = func(x_e)
                        evals_used += 1
                        
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_vals[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_vals[-1] = f_r
                    elif f_r < f_vals[-2]:
                        # Accept reflection
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                    else:
                        # Contraction
                        # External contraction
                        x_c = centroid + rho * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, -5.0, 5.0)
                        f_c = func(x_c)
                        evals_used += 1
                        
                        if f_c < f_vals[-1]:
                            simplex[-1] = x_c
                            f_vals[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                f_new = func(simplex[i])
                                f_vals[i] = f_new
                                evals_used += 1
                                if evals_used >= budget:
                                    break
                            if evals_used >= budget:
                                break
                    
                    # Update local best
                    idx_min = np.argmin(f_vals)
                    if f_vals[idx_min] < local_best_f:
                        local_best_f = f_vals[idx_min]
                        local_best_x = simplex[idx_min].copy()
                        
                    # Update global best
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()

            return best_x, best_f
