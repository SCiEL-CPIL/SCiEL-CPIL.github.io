# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using Tchebycheff scalarization + Simulated Annealing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _tchebycheff_scalarize(self, f, weights, zmin, zmax):
        """Compute Tchebycheff scalarization value."""
        # Normalize objectives
        f_norm = (f - zmin) / (zmax - zmin + 1e-10)
        # Tchebycheff: max_i weights[i] * f_norm[i]
        return np.max(weights * f_norm)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        
        # Generate deterministic grid of Tchebycheff weights (simplex points)
        # For 3 objectives, use a grid on the simplex
        n_weights = 50
        weights_list = []
        
        # Create points on the 2D simplex for 3 objectives
        # Use a combination of vertices and edge midpoints and interior points
        # Vertices: (1,0,0), (0,1,0), (0,0,1)
        # Edge midpoints: (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
        # Interior points: distribute remaining
        
        # First add vertices and edge midpoints
        base_weights = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.333, 0.333, 0.334],
        ]
        weights_list.extend(base_weights)
        
        # Add more points to reach n_weights
        remaining = n_weights - len(weights_list)
        if remaining > 0:
            for _ in range(remaining):
                # Random point on simplex
                u = rng.random(3)
                u = u / u.sum()
                weights_list.append(u.tolist())
        
        weights_arr = np.array(weights_list[:n_weights])
        
        # Evaluate initial points using Tchebycheff optimization
        # For each weight, do a few SA iterations to find a good point
        X_init = np.zeros((n_weights, dim))
        F_init = np.zeros((n_weights, n_obj))
        
        used = 0
        zmin = np.full(n_obj, np.inf)
        zmax = np.full(n_obj, -np.inf)
        
        # First pass: get initial estimates of zmin/zmax using random points
        n_init_random = min(10, budget - n_weights)
        for i in range(n_init_random):
            if used >= budget:
                break
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            used += 1
            zmin = np.minimum(zmin, f)
            zmax = np.maximum(zmax, f)
        
        # Now for each weight, run a short SA to find a good point
        for w_idx in range(n_weights):
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            w = weights_arr[w_idx]
            
            # Start from a random point
            x_curr = lo + (hi - lo) * rng.random(dim)
            f_curr = problem(x_curr)
            used += 1
            zmin = np.minimum(zmin, f_curr)
            zmax = np.maximum(zmax, f_curr)
            
            # SA for this weight: 3-5 iterations
            n_sa = 4
            T = 0.1
            T_min = 0.001
            step_scale = 0.3
            
            for sa_iter in range(n_sa):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Generate neighbor
                x_new = x_curr + step_scale * (hi - lo) * rng.standard_normal(dim)
                x_new = np.clip(x_new, lo, hi)
                f_new = problem(x_new)
                used += 1
                zmin = np.minimum(zmin, f_new)
                zmax = np.maximum(zmax, f_new)
                
                # Compute scalarization values
                zrange = zmax - zmin + 1e-10
                curr_val = np.max(w * (f_curr - zmin) / zrange)
                new_val = np.max(w * (f_new - zmin) / zrange)
                
                delta = new_val - curr_val
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    x_curr = x_new
                    f_curr = f_new
                
                # Cool down
                T *= 0.5
                if T < T_min:
                    T = T_min
            
            X_init[w_idx] = x_curr
            F_init[w_idx] = f_curr
        
        # Now we have n_weights points. Use remaining budget for further refinement.
        # Build a population from these points and do more SA iterations on the best ones.
        
        # Keep track of all evaluated points
        all_F = F_init.copy()
        all_X = X_init.copy()
        
        # Select top points based on hypervolume contribution (approximate)
        # For simplicity, refine all points further with more SA iterations
        
        remaining_budget = budget - used
        if remaining_budget > 0:
            # Do more SA iterations on each point
            iters_per_point = max(1, remaining_budget // n_weights)
            
            for w_idx in range(n_weights):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                w = weights_arr[w_idx]
                x_curr = all_X[w_idx]
                f_curr = all_F[w_idx]
                
                T = 0.05
                T_min = 0.0005
                step_scale = 0.15
                
                for sa_iter in range(iters_per_point):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Generate neighbor with decreasing step
                    current_step = step_scale * (0.8 ** sa_iter)
                    x_new = x_curr + current_step * (hi - lo) * rng.standard_normal(dim)
                    x_new = np.clip(x_new, lo, hi)
                    f_new = problem(x_new)
                    used += 1
                    zmin = np.minimum(zmin, f_new)
                    zmax = np.maximum(zmax, f_new)
                    
                    # Compute scalarization values
                    zrange = zmax - zmin + 1e-10
                    curr_val = np.max(w * (f_curr - zmin) / zrange)
                    new_val = np.max(w * (f_new - zmin) / zrange)
                    
                    delta = new_val - curr_val
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        x_curr = x_new
                        f_curr = f_new
                        all_X[w_idx] = x_curr
                        all_F[w_idx] = f_new
                    
                    # Cool down
                    T *= 0.7
                    if T < T_min:
                        T = T_min
                
                # Record final point
                all_F = np.vstack([all_F, f_curr])
        
        return all_X
# EVOLVE-BLOCK-END
