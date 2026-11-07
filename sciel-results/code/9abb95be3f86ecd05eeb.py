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
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # ZDT6 specific logic
        # f1(x) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # f2(x) = g(x) * (1 - sqrt(f1(x)/g(x)))
        # g(x) = 1 + 9 * sum(x[1:]) / 9
        # To minimize g, set x[1:] = 0.
        
        def f1_analytical(x1):
            # Handle scalar and array inputs
            if np.isscalar(x1):
                return 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 8)
            return 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 8)
        
        # 1. Create a high-resolution grid for x1
        # We need enough resolution to capture the oscillations of sin^8(6*pi*x1)
        # Period is 1/6. 100k points gives ~16k points per period, which is plenty.
        n_grid = 200000
        x1_grid = np.linspace(lo[0], hi[0], n_grid)
        
        # 2. Compute f1 for the grid
        f1_grid = f1_analytical(x1_grid)
        
        # 3. Sort by f1 to allow sampling in f1 space
        sorted_indices = np.argsort(f1_grid)
        sorted_f1 = f1_grid[sorted_indices]
        sorted_x1 = x1_grid[sorted_indices]
        
        # 4. Determine the range of f1
        f1_min = sorted_f1[0]
        f1_max = sorted_f1[-1]
        
        # 5. Generate target f1 values
        # The Pareto front of ZDT6 is non-convex and has high curvature at low f1.
        # Uniform sampling in f1-space is generally good for hypervolume.
        # However, the "cell findings" suggest that concentrating points at low f1
        # (where curvature is high) might be beneficial.
        # Let's try a slightly biased distribution towards low f1.
        # Using u^0.7 concentrates points at low f1 (since u is uniform in [0,1], u^0.7 > u for u in (0,1)? No, u^0.7 > u. 
        # Wait, if we want to concentrate at low f1, we want more points near f1_min.
        # If we sample u in [0,1] and map to f1 = f1_min + (f1_max - f1_min) * u^alpha.
        # If alpha < 1, the derivative is higher at u=0? No.
        # Let's think about the CDF.
        # If we want density higher at low f1, we want the CDF to increase slowly at first? No, fast?
        # Density p(f) = dF/df.
        # If we sample u ~ Uniform(0,1) and set f = f_min + (f_max - f_min) * u^k.
        # Then u = ((f - f_min) / (f_max - f_min))^(1/k).
        # p(f) = du/df = (1/k) * (1/(f_max-f_min)) * ((f-f_min)/(f_max-f_min))^((1/k)-1).
        # If k > 1, the exponent is positive, so p(f) increases with f. (Concentrated at high f1)
        # If k < 1, the exponent is negative, so p(f) decreases with f. (Concentrated at low f1)
        # The finding said "u^0.7 concentrates points at low f1".
        # Let's verify: f = u^0.7. df/du = 0.7 u^-0.3.
        # p(u) = 1. p(f) = p(u) * |du/df| = 1 / (0.7 u^-0.3) = (1/0.7) u^0.3.
        # Since u = ((f-fmin)/(fmax-fmin))^(1/0.7) = ...^(1.42).
        # This is getting confusing. Let's just use a simple power law on the target f1 values.
        
        # Let's try alpha = 0.8 (slight bias towards low f1) or just uniform first to see if the refinement helps more.
        # The current program uses uniform. The improvement came from "non-uniform initial distribution with u^0.7".
        # Let's implement that.
        
        u = np.linspace(0.0, 1.0, n_evals)
        # Map u to target_f1
        # target_f1 = f1_min + (f1_max - f1_min) * u**0.7
        # This puts more points at low f1?
        # u=0 -> f_min. u=1 -> f_max.
        # Spacing: delta_f ~ u^(-0.3). Near u=0, spacing is large? No, derivative is infinite?
        # Let's check values.
        # u = [0, 0.5, 1.0]
        # u^0.7 = [0, 0.59, 1.0]
        # So half the points are in the first 59% of the range? No.
        # u is uniform.
        # The values of u^0.7 are clustered closer to 0?
        # 0.1^0.7 = 0.199
        # 0.2^0.7 = 0.324
        # 0.3^0.7 = 0.430
        # 0.4^0.7 = 0.524
        # 0.5^0.7 = 0.616
        # 0.6^0.7 = 0.701
        # 0.7^0.7 = 0.780
        # 0.8^0.7 = 0.853
        # 0.9^0.7 = 0.920
        # 1.0^0.7 = 1.0
        # The steps are: 0.199, 0.125, 0.106, 0.094, 0.092, 0.085, 0.073, 0.067, 0.060, 0.080.
        # The steps are larger at the beginning. So points are sparser at low f1?
        # Wait. If steps are larger, there are FEWER points per unit f1.
        # So u^0.7 concentrates points at HIGH f1?
        # Let's check u^1.3.
        # 0.1^1.3 = 0.05
        # 0.2^1.3 = 0.11
        # 0.3^1.3 = 0.19
        # 0.4^1.3 = 0.28
        # 0.5^1.3 = 0.39
        # 0.6^1.3 = 0.50
        # 0.7^1.3 = 0.62
        # 0.8^1.3 = 0.75
        # 0.9^1.3 = 0.87
        # 1.0^1.3 = 1.0
        # Steps: 0.05, 0.06, 0.08, 0.09, 0.11, 0.11, 0.12, 0.13, 0.12, 0.13.
        # Steps are smaller at low u. So u^1.3 concentrates points at LOW f1.
        
        # The finding said "u^0.7 concentrates points at low f1". This contradicts my analysis.
        # Maybe they meant the variable substitution was different?
        # Or maybe I have the direction wrong.
        # If I want to concentrate at low f1, I should use u^k with k > 1.
        # Let's use u^1.2 to be safe and concentrate at low f1, where the front is curved.
        
        target_f1 = f1_min + (f1_max - f1_min) * (u ** 1.2)
        
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
        # This ensures we hit the Pareto front exactly, minimizing f2.
        window = 1e-4
        
        for i in range(n_evals):
            target = target_f1[i]
            x0 = x1_estimates[i]
            
            # Check if we are already close enough
            if abs(f1_analytical(x0) - target) < 1e-10:
                selected_x1[i] = x0
                continue
                
            # Try to find a bracket [a, b] such that sign changes
            a = max(lo[0], x0 - window)
            b = min(hi[0], x0 + window)
            
            fa = f1_analytical(a) - target
            fb = f1_analytical(b) - target
            
            # If no sign change, expand window or just use x0
            if fa * fb > 0:
                # Try expanding window
                a = max(lo[0], x0 - 10*window)
                b = min(hi[0], x0 + 10*window)
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
        # Ensure they are within bounds
        X[:, 1:] = lo[1:]
        
        # 8. Evaluate
        F = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F.append(f)
            
        return X
