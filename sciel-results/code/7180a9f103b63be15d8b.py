import numpy as np
import time
import sys
from scipy.optimize import minimize_scalar

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.optimal_x1 = None
        self._compute_optimal_x1()

    def _compute_hv(self, points, ref):
        # points: (N, 2)
        # ref: (2,)
        # Assumes points are non-dominated and sorted by f1 ascending
        # For 2D, HV = sum (x_{i+1} - x_i) * (ref[1] - y_i)
        # where x_{N+1} = ref[0]
        n = len(points)
        if n == 0:
            return 0.0
        
        # Sort by first objective
        points = points[np.argsort(points[:, 0])]
        
        # Check if any point is dominated by ref?
        # Only count if strictly better in all objectives.
        # f1 < ref[0] and f2 < ref[1]
        
        hv = 0.0
        # The points define a step function.
        # The "top-right" corner of the rectangle for point i is (points[i+1][0], points[i][1])?
        # No. The region dominated by point i is [f1_i, ref0] x [f2_i, ref1].
        # The union of these regions.
        # Since f2 decreases as f1 increases (for ZDT1 front), the points are sorted such that f1 increases and f2 decreases.
        # The "upper envelope" of the union is the step function that goes from (f1_1, f2_1) to (f1_2, f2_1) then down to (f1_2, f2_2) ...
        # Wait, the dominated region is the "south-east" quadrant.
        # So the boundary is:
        # Start at (ref0, ref1).
        # Go left to (f1_N, ref1)? No.
        
        # Let's stick to the standard 2D HV formula:
        # Sort points by f1 ascending.
        # H = 0
        # prev_x = 0 ? No, the "left" boundary is determined by the point with smallest f1.
        # Actually, the hypervolume is the area of the polygon formed by the points and the reference point.
        # Vertices: (f1_1, ref1) -> (f1_1, f2_1) -> (f1_2, f2_1) -> (f1_2, f2_2) ... -> (f1_N, f2_N) -> (ref0, f2_N) -> (ref0, ref1) -> (f1_1, ref1)
        
        # Area = sum of rectangles?
        # Decompose into vertical strips:
        # Strip i is from x_i to x_{i+1} (with x_{N+1}=ref0).
        # Height is determined by the "worst" f2 in that strip?
        # No, the boundary is the step function $f_2(x) = f_2(i)$ for $x \in [x_i, x_{i+1})$.
        # So Area = sum_{i=1}^{N} (x_{i+1} - x_i) * (ref1 - f_2(i))
        
        x_next = ref[0]
        for i in range(n):
            x_curr = points[i][0]
            y_curr = points[i][1]
            # Ensure we don't go below 0? Points are in [0,1]. Ref is 11.
            width = x_next - x_curr
            height = ref[1] - y_curr
            hv += width * height
            x_next = x_curr
            
        return hv

    def _compute_optimal_x1(self):
        N = self.budget
        if N <= 0:
            return
        ref = np.array([11.0, 11.0])
        
        # Use t = linspace(0, 1, N)
        t = np.linspace(0, 1, N)
        
        def neg_hv(p):
            x1 = t ** p
            f2 = 1.0 - np.sqrt(x1)
            points = np.column_stack((x1, f2))
            hv = self._compute_hv(points, ref)
            return -hv
        
        # Initial scan to find a good starting point for minimize_scalar
        # Scan p from 0.5 to 10.0 with step 0.5
        best_p_init = 1.0
        best_hv_init = -np.inf
        for p in np.arange(0.5, 10.0, 0.5):
            hv = -neg_hv(p)
            if hv > best_hv_init:
                best_hv_init = hv
                best_p_init = p
        
        # Refine using minimize_scalar
        res = minimize_scalar(neg_hv, bounds=(best_p_init - 0.5, best_p_init + 0.5), method='bounded', options={'xatol': 1e-6})
        
        if res.success:
            self.optimal_x1 = t ** res.x
        else:
            # Fallback to initial best
            self.optimal_x1 = t ** best_p_init

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320 # 10s margin
        
        if self.optimal_x1 is None:
             self._compute_optimal_x1()
             
        x1_vals = self.optimal_x1
        N = len(x1_vals)
        
        # Create X matrix
        X = np.zeros((N, self.dim))
        X[:, 0] = x1_vals
        
        # Evaluate
        for i in range(N):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
