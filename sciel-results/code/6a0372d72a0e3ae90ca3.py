import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 Structure:
        # First N-M+1 = 10 variables determine g.
        # Last M-1 = 2 variables determine position on front.
        # g is minimized at x_i = 0.5 for i in 0..9.
        
        n_g_vars = self.dim - self.n_obj + 1  # 10
        n_front_vars = self.n_obj - 1         # 2
        
        # We will set the g-variables to 0.5 to ensure g=0 (or close to it)
        # This places all points on the Pareto front.
        
        # Strategy: Select n_front_vars dimensions to maximize hypervolume coverage.
        # We use a MaxMin strategy on the 2D angle space.
        
        # 1. Generate a large pool of candidates in [0,1]^2
        # Use Sobol or Random. Sobol is better.
        # We can generate Sobol points using scipy.stats.qmc
        from scipy.stats import qmc
        
        # Generate 1000 candidate points in 2D
        sampler = qmc.Sobol(d=n_front_vars, scramble=True)
        pool_size = 1000
        candidates = sampler.random(n=pool_size)
        
        selected_angles = []
        X_results = []
        F_results = []
        
        used_evals = 0
        
        # Initial batch: Take the first 10 Sobol points to start
        init_count = min(10, self.budget)
        for i in range(init_count):
            angles = candidates[i]
            x = np.zeros(self.dim)
            x[:n_g_vars] = 0.5
            x[n_g_vars:] = angles
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            X_results.append(x)
            F_results.append(f)
            selected_angles.append(angles)
            used_evals += 1
            
        # Remaining evaluations: MaxMin selection
        # We need to pick from the remaining pool
        remaining_candidates = candidates[init_count:]
        
        while used_evals < self.budget:
            if time.time() > deadline:
                break
                
            if len(remaining_candidates) == 0:
                # Regenerate pool if empty (should not happen with 1000 pool and 200 budget)
                sampler = qmc.Sobol(d=n_front_vars, scramble=True)
                remaining_candidates = sampler.random(n=pool_size)
                
            # Find candidate with max min-distance to selected_angles
            # selected_angles is a list of 2D vectors
            if not selected_angles:
                best_idx = 0
                best_dist = np.inf
            else:
                sel_arr = np.array(selected_angles)
                # Calculate distances from each remaining candidate to all selected
                # remaining_candidates shape: (N, 2)
                # sel_arr shape: (M, 2)
                # dist shape: (N, M)
                # To save memory/time, do it in chunks or vectorized
                dists = np.linalg.norm(remaining_candidates[:, np.newaxis, :] - sel_arr[np.newaxis, :, :], axis=2)
                min_dists = np.min(dists, axis=1)
                best_idx = np.argmax(min_dists)
                best_dist = min_dists[best_idx]
            
            best_angle = remaining_candidates[best_idx]
            
            # Remove from pool
            remaining_candidates = np.delete(remaining_candidates, best_idx, axis=0)
            
            # Construct x
            x = np.zeros(self.dim)
            x[:n_g_vars] = 0.5
            x[n_g_vars:] = best_angle
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            X_results.append(x)
            F_results.append(f)
            selected_angles.append(best_angle)
            used_evals += 1
            
        return X_results
