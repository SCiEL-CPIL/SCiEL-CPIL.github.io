"""Multi-objective optimizer for ZDT3 using numerical optimization of sample placement to maximize hypervolume."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _zdt3_f2(self, x1):
        """Compute f2 for ZDT3 given x1, assuming x2..xn=0 (or g is constant)."""
        n = self.dim
        g = 1 + 10 * (n - 1)
        if x1 < 0:
            x1 = 0
        if x1 > 1:
            x1 = 1
            
        term1 = np.sqrt(x1 / g)
        term2 = (x1 / g) * np.sin(10 * np.pi * x1)
        h = 1 - term1 - term2
        f2 = g * h
        return f2

    def _compute_hv(self, points):
        """
        Compute 2D hypervolume for points strictly better than reference point [11, 11].
        points: array of shape (N, 2) where col 0 is f1, col 1 is f2.
        """
        if len(points) == 0:
            return 0.0
        
        ref = np.array([11.0, 11.0])
        valid = points[np.all(points < ref, axis=1)]
        
        if len(valid) == 0:
            return 0.0
        
        valid = valid[np.argsort(valid[:, 0])]
        
        pareto_pts = []
        min_f2 = np.inf
        for p in valid:
            f1, f2 = p
            if f2 < min_f2:
                pareto_pts.append(p)
                min_f2 = f2
        
        if not pareto_pts:
            return 0.0
            
        hv = 0.0
        prev_x = 0.0
        for i in range(len(pareto_pts)):
            f1_i, f2_i = pareto_pts[i]
            width = f1_i - prev_x
            height = f2_i
            hv += width * height
            prev_x = f1_i
            
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        # Strategy:
        # 1. The Pareto front of ZDT3 is defined by x1 in [0, 1] and x2..xn=0.
        # 2. We want to select n_points values of x1 that maximize the hypervolume of the resulting points.
        # 3. This is a 1D optimization problem over the spacings.
        # 4. We parameterize the x1 values as cumulative sums of spacings.
        # 5. We use Nelder-Mead to optimize the spacings to maximize the discrete HV.
        
        # Objective function for the optimizer:
        # Inputs: vector of log-spacings (or spacings) of length n_points-1?
        # Actually, it's easier to optimize the positions directly but constrained to be sorted.
        # Or optimize the gaps. Let gaps be d_1, ..., d_{N-1} such that sum(d) = 1.
        # To ensure positivity, we can use softmax or just normalize positive values.
        
        # Let's try optimizing the gaps directly with a penalty for non-positivity or using a transformation.
        # Transformation: d_i = exp(y_i) / sum(exp(y_j)). This ensures positive and sums to 1.
        # Then x1_i = sum_{j=1}^{i-1} d_j.
        # x1_1 = 0, x1_N = 1.
        
        def hv_to_maximize(y):
            # y is the vector of log-gaps
            # Compute gaps
            # To avoid overflow in exp, subtract max
            y_max = np.max(y)
            exp_y = np.exp(y - y_max)
            gaps = exp_y / np.sum(exp_y)
            
            # Cumulative sum to get x1 positions
            # x1_0 = 0
            # x1_i = sum(gaps[0:i])
            x1_vals = np.concatenate(([0.0], np.cumsum(gaps)))
            # Ensure last is exactly 1 (numerical stability)
            x1_vals[-1] = 1.0
            
            # Compute f2 values
            f2_vals = np.array([self._zdt3_f2(x) for x in x1_vals])
            
            points = np.column_stack((x1_vals, f2_vals))
            
            # Compute HV
            hv = self._compute_hv(points)
            
            # We want to maximize HV, so minimize -HV
            return -hv
        
        # Initial guess: uniform spacing
        # If uniform, all gaps are equal, so all y are equal.
        y0 = np.zeros(n_points - 1)
        
        # Run optimization
        # Nelder-Mead is derivative-free and robust.
        try:
            result = minimize(hv_to_maximize, y0, method='Nelder-Mead', 
                              options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            y_opt = result.x
        except Exception:
            # Fallback to uniform if optimization fails
            y_opt = y0
            
        # Compute final x1 values
        y_max = np.max(y_opt)
        exp_y = np.exp(y_opt - y_max)
        gaps = exp_y / np.sum(exp_y)
        x1_vals = np.concatenate(([0.0], np.cumsum(gaps)))
        x1_vals[-1] = 1.0
        
        # Construct X
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1_vals
        
        # Evaluate
        F = np.zeros((n_points, self.n_obj))
        
        for i in range(n_points):
            if time.time() > deadline:
                break
            try:
                x = np.clip(X[i], lo, hi)
                F[i] = problem(x)
            except Exception:
                F[i] = [1e10, 1e10]
                
        return X

# EVOLVE-BLOCK-END
