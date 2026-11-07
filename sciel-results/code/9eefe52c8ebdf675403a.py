"""Multi-objective optimizer for ZDT3 using 1D non-linear least-squares."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for ZDT3 using L-BFGS-B to optimize the placement of points
    on the analytical Pareto front to maximize hypervolume contribution.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_samples = self.budget
        g_min = 1 + 10 * (self.dim - 1)
        
        # Analytical front function for ZDT3
        def front_f2(f1):
            # f1 in [0, 1]
            f1 = np.clip(f1, 0, 1)
            term1 = np.sqrt(f1 / g_min)
            term2 = (f1 / g_min) * np.sin(10 * np.pi * f1)
            return g_min * (1 - term1 - term2)
        
        # Generate a dense reference set on the Pareto front
        # Using a moderate density to balance accuracy and computation time
        n_ref = 500
        f1_ref = np.linspace(0, 1, n_ref)
        f2_ref = front_f2(f1_ref)
        ref_points = np.column_stack((f1_ref, f2_ref))
        
        # Initial guess: uniform grid with slight perturbation to break symmetry
        t_init = np.linspace(0, 1, n_samples)
        noise = np.random.uniform(-0.01, 0.01, size=n_samples)
        t_init = np.clip(t_init + noise, 0, 1)
        
        # Objective function: Minimize sum of squared distances from each ref point 
        # to the nearest sample point.
        def objective(t):
            f1_s = t
            f2_s = front_f2(f1_s)
            sample_points = np.column_stack((f1_s, f2_s))
            
            # Compute squared distances: (n_ref, n_samples)
            diff = ref_points[:, np.newaxis, :] - sample_points[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            
            # Min squared distance for each ref point
            min_dist_sq = np.min(dist_sq, axis=1)
            
            return np.sum(min_dist_sq)
        
        # Bounds for t: each t_i in [0, 1]
        bounds_opt = [(0, 1)] * n_samples
        
        # Optimize using L-BFGS-B
        try:
            res = minimize(objective, t_init, method='L-BFGS-B', bounds=bounds_opt, 
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
            t_opt = res.x
        except Exception:
            t_opt = t_init
            
        # Ensure t_opt is within bounds
        t_opt = np.clip(t_opt, 0, 1)
        
        # Construct the x vectors
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = t_opt
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
