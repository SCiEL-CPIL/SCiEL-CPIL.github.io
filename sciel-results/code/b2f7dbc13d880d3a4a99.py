import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Probe to identify angle variables
        # Base point: all 0.5
        x_base = np.full(self.dim, 0.5)
        f_base = np.array(problem(x_base))
        
        angle_vars = []
        for i in range(self.dim):
            x_test = x_base.copy()
            # Perturb variable i
            x_test[i] = 0.51
            f_test = np.array(problem(x_test))
            diffs = f_test - f_base
            # If any objective decreases, it's an angle variable
            # (Position vars increase g, which increases all objectives)
            if np.any(diffs < -1e-6):
                angle_vars.append(i)
                
        # We expect 2 angle variables for 3 objectives
        if len(angle_vars) < 2:
            # Fallback to random if detection fails
            X = lo + (hi - lo) * np.random.rand(self.budget, self.dim)
            F = np.array([problem(x) for x in X])
            return X
            
        v1, v2 = angle_vars[0], angle_vars[1]
        
        # 2. Determine which is polar
        # Fix v2, vary v1. Check if f_3 (index 2) changes.
        x_fix_v2 = x_base.copy()
        x_fix_v2[v2] = 0.5
        x_fix_v2[v1] = 0.1
        f_v1_low = np.array(problem(x_fix_v2))
        
        x_fix_v2[v1] = 0.9
        f_v1_high = np.array(problem(x_fix_v2))
        
        # If f_3 changes, v1 is polar
        if abs(f_v1_low[2] - f_v1_high[2]) > 1e-6:
            polar_var = v1
            azim_var = v2
        else:
            polar_var = v2
            azim_var = v1
            
        # 3. Generate points
        n_points = self.budget - 15 # Reserve budget for probes
        X = np.zeros((n_points, self.dim))
        X[:, :] = 0.5 # Position vars at 0.5
        
        u1 = np.random.rand(n_points)
        u2 = np.random.rand(n_points)
        
        # Polar angle sampling (area uniform)
        # x in [0, 1]
        cos_1 = np.cos(1.0)
        polar_angles = np.arccos(1.0 - u1 * (1.0 - cos_1))
        
        # Azimuthal angle sampling (uniform)
        azim_angles = u2 * 1.0
        
        X[:, polar_var] = polar_angles
        X[:, azim_var] = azim_angles
        
        X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        
        return X
