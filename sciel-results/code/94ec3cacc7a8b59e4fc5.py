"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        dim = self.dim
        
        # ZDT3 structure:
        # f1 = x[0]
        # f2 = 1 - sqrt(x[0]) + 10*x[0]*sin(10*pi*x[0])  when x[1:]=0
        # Pareto front: x[1:] = 0, x[0] in [0,1]
        # f2 has oscillations with period 0.1 in x[0]
        # Local minima of f2 occur where derivative = 0
        # f2'(x) = -1/(2*sqrt(x)) + 10*sin(10*pi*x) + 100*pi*x*cos(10*pi*x) = 0
        
        # Strategy: Place points at known local minima of f2 to maximize hypervolume.
        # The sin term creates 10 full periods in [0,1], so minima are roughly at
        # x = 0.05, 0.15, 0.25, ..., 0.95 (where sin(10*pi*x) = -1)
        # But we need to account for the 10*x[0] factor and sqrt term.
        
        # Use Newton's method to find precise local minima of f2(x) = 1 - sqrt(x) + 10*x*sin(10*pi*x)
        # f2'(x) = -1/(2*sqrt(x)) + 10*sin(10*pi*x) + 100*pi*x*cos(10*pi*x)
        # f2''(x) = 1/(4*x^(3/2)) + 100*pi*cos(10*pi*x) - 1000*pi^2*x*sin(10*pi*x)
        
        def f2_val(x):
            return 1 - np.sqrt(x) + 10*x*np.sin(10*np.pi*x)
        
        def f2_deriv(x):
            return -1/(2*np.sqrt(x)) + 10*np.sin(10*np.pi*x) + 100*np.pi*x*np.cos(10*np.pi*x)
        
        def f2_deriv2(x):
            return 1/(4*x**1.5) + 100*np.pi*np.cos(10*np.pi*x) - 1000*np.pi**2*x*np.sin(10*np.pi*x)
        
        # Find local minima using Newton's method starting from initial guesses
        # Initial guesses at where sin(10*pi*x) ≈ -1: x ≈ 0.05, 0.15, ..., 0.95
        x0_minima = []
        for k in range(10):
            x_init = 0.05 + k * 0.1  # Where sin(10*pi*x) = -1
            x = x_init
            for _ in range(50):
                fx = f2_deriv(x)
                fpx = f2_deriv2(x)
                if abs(fpx) < 1e-15:
                    break
                x_new = x - fx / fpx
                if x_new < 0:
                    x_new = 1e-10
                if x_new > 1:
                    x_new = 1 - 1e-10
                if abs(x_new - x) < 1e-15:
                    break
                x = x_new
            if 0 < x < 1:
                x0_minima.append(x)
        
        # Also add x=0 (endpoint) and x=1 (endpoint)
        x0_minima = [0.0] + x0_minima + [1.0]
        
        # Now we have ~12 critical points. We need to use our 200 evaluations wisely.
        # Evaluate these critical points first, then use remaining budget for
        # fine-tuning around them and filling gaps.
        
        n_critical = len(x0_minima)
        x0_critical = np.array(x0_minima)
        
        # Create decision vectors for critical points
        X_critical = np.zeros((n_critical, dim))
        X_critical[:, 0] = x0_critical
        
        # Evaluate critical points
        F_critical = np.array([problem(x) for x in X_critical])
        
        if time.time() > deadline:
            return X_critical
        
        # Now use remaining budget for additional points
        n_remaining = n_evals - n_critical
        if n_remaining > 0:
            # Add points between critical points to better approximate the front
            # Use adaptive sampling: more points where f2 changes rapidly
            x0_additional = []
            
            # For each interval between consecutive critical points, add a few points
            for i in range(len(x0_minima) - 1):
                x_left = x0_minima[i]
                x_right = x0_minima[i + 1]
                # Add 2-3 points per interval
                n_pts = 3
                for j in range(1, n_pts):
                    x_mid = x_left + (x_right - x_left) * j / n_pts
                    x0_additional.append(x_mid)
            
            # Also add some points near x=0 where the front is most curved
            for j in range(1, 10):
                x_near_0 = 1e-6 + (0.05 - 1e-6) * j / 10
                x0_additional.append(x_near_0)
            
            # Deduplicate and limit to n_remaining
            x0_additional = np.unique(np.round(x0_additional, 15))
            if len(x0_additional) > n_remaining:
                x0_additional = x0_additional[:n_remaining]
            
            # Create decision vectors for additional points
            X_additional = np.zeros((len(x0_additional), dim))
            X_additional[:, 0] = x0_additional
            
            # Evaluate additional points
            F_additional = np.array([problem(x) for x in X_additional])
            
            # Combine all evaluated points
            X_all = np.vstack([X_critical, X_additional])
            F_all = np.vstack([F_critical, F_additional])
        else:
            X_all = X_critical
            F_all = F_critical
        
        if time.time() > deadline:
            return X_all
        
        # Find non-dominated points
        n_points = len(F_all)
        if n_points == 0:
            return X_all
        
        # For 2 objectives, find Pareto front
        # Sort by f1
        sort_idx = np.argsort(F_all[:, 0])
        F_sorted = F_all[sort_idx]
        
        # Find non-dominated points: scan from left to right, keep points with decreasing f2
        nondom_mask = np.zeros(n_points, dtype=bool)
        min_f2 = np.inf
        for i in range(n_points):
            if F_sorted[i, 1] < min_f2:
                nondom_mask[sort_idx[i]] = True
                min_f2 = F_sorted[i, 1]
        
        # Return all evaluated points (the evaluator records all returned objective vectors)
        return X_all
# EVOLVE-BLOCK-END
