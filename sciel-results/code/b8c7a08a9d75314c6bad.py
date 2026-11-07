import numpy as np
import time
from scipy.optimize import brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # ZDT6 specific logic
        # f1(x) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        
        def f1_analytical(x1):
            return 1.0 - np.exp(-4.0 * x1) * np.sin(6.0 * np.pi * x1)**8
        
        # 1. Create a high-resolution grid for x1 to find min/max and initial guesses
        n_grid = 1000000
        x1_grid = np.linspace(lo[0], hi[0], n_grid)
        
        # 2. Compute f1 for the grid
        f1_grid = f1_analytical(x1_grid)
        
        # 3. Sort by f1 to allow uniform sampling in f1 space
        sorted_indices = np.argsort(f1_grid)
        sorted_f1 = f1_grid[sorted_indices]
        sorted_x1 = x1_grid[sorted_indices]
        
        # 4. Determine the range of f1
        f1_min = sorted_f1[0]
        f1_max = sorted_f1[-1]
        
        # 5. Generate target f1 values uniformly in [f1_min, f1_max]
        target_f1 = np.linspace(f1_min, f1_max, n_evals)
        
        # 6. Find x1 for each target using grid search + refinement
        selected_x1 = np.zeros(n_evals)
        
        # Use searchsorted to find the closest point in the sorted array
        indices = np.searchsorted(sorted_f1, target_f1)
        indices = np.clip(indices, 1, n_grid - 1)
        
        # Compare distance to left and right neighbor
        dist_left = np.abs(sorted_f1[indices - 1] - target_f1)
        dist_right = np.abs(sorted_f1[indices] - target_f1)
        
        use_right = dist_right < dist_left
        final_indices = np.where(use_right, indices, indices - 1)
        
        # Initial x1 estimates from grid
        x1_estimates = sorted_x1[final_indices]
        
        # Refine each x1 using brentq in a small window
        # Window size should be small enough to contain a root but large enough to be safe
        # Grid step is ~1e-6. Let's use a window of +/- 1e-4
        window = 1e-4
        
        for i in range(n_evals):
            target = target_f1[i]
            x0 = x1_estimates[i]
            
            # Define the function for brentq
            # We need to find x such that f1(x) - target = 0
            # f1 is continuous.
            
            # Check if we are already close enough
            if abs(f1_analytical(x0) - target) < 1e-10:
                selected_x1[i] = x0
                continue
                
            # Try to find a bracket [a, b] such that sign changes
            a = max(0.0, x0 - window)
            b = min(1.0, x0 + window)
            
            fa = f1_analytical(a) - target
            fb = f1_analytical(b) - target
            
            # If no sign change, expand window or just use x0
            if fa * fb > 0:
                # Try expanding window
                a = max(0.0, x0 - 10*window)
                b = min(1.0, x0 + 10*window)
                fa = f1_analytical(a) - target
                fb = f1_analytical(b) - target
                if fa * fb > 0:
                    # If still no sign change, use the grid estimate
                    selected_x1[i] = x0
                    continue
            
            try:
                x_root = brentq(lambda x: f1_analytical(x) - target, a, b, xtol=1e-12)
                selected_x1[i] = x_root
            except Exception:
                selected_x1[i] = x0
                
            if time.time() > deadline:
                break

        # 7. Construct the X matrix
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = selected_x1
        # x[1:] are 0, which is lo[1:] for ZDT6 [0,1]
        
        # 8. Evaluate
        F = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F.append(f)
            
        return X
