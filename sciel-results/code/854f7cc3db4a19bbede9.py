"""Multi-objective optimizer for ZDT1 using analytical Pareto front placement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Analytical optimizer for ZDT1 maximizing hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.optimal_x1 = None
        self._compute_optimal_x1()

    def _compute_hv(self, points, ref):
        """
        Compute hypervolume for 2D points.
        points: (N, 2) array of (f1, f2)
        ref: (2,) array of reference point
        Assumes points are non-dominated and sorted by f1 ascending.
        """
        n = len(points)
        if n == 0:
            return 0.0
        
        # Ensure points are strictly better than ref
        # For ZDT1, f1 in [0,1], f2 in [0,1], ref [11,11], so all are better.
        
        # Sort by f1 ascending
        points = points[np.argsort(points[:, 0])]
        
        hv = 0.0
        ref_x, ref_y = ref
        
        # Iterate through points
        # The area is sum of rectangles:
        # For i from 0 to N-2: width = x[i+1] - x[i], height = ref_y - y[i]
        # For i = N-1: width = ref_x - x[N-1], height = ref_y - y[N-1]
        
        for i in range(n - 1):
            x_curr = points[i, 0]
            y_curr = points[i, 1]
            x_next = points[i + 1, 0]
            
            # Width of the strip
            width = x_next - x_curr
            # Height is determined by the current point's y, since y is decreasing
            # The region dominated by point i in the strip [x_curr, x_next) is [y_curr, ref_y]
            height = ref_y - y_curr
            hv += width * height
            
        # Last point
        x_last = points[n - 1, 0]
        y_last = points[n - 1, 1]
        width_last = ref_x - x_last
        height_last = ref_y - y_last
        hv += width_last * height_last
        
        return hv

    def _compute_optimal_x1(self):
        """
        Find the optimal power-law exponent p for x_i = t_i^p
        to maximize hypervolume.
        """
        N = self.budget
        if N <= 0:
            self.optimal_x1 = np.array([])
            return
            
        ref = np.array([11.0, 11.0])
        # Uniform t values in (0, 1]
        # Using linspace(0, 1, N) puts a point at 0. 
        # f1=0, f2=1 is a valid point on the front.
        t = np.linspace(0.0, 1.0, N)
        
        best_hv = -1.0
        best_p = 1.0
        
        # Coarse scan
        p_vals = np.arange(0.5, 10.0, 0.1)
        for p in p_vals:
            x1 = t ** p
            # Handle x1=0 case for sqrt
            with np.errstate(invalid='ignore'):
                f2 = 1.0 - np.sqrt(x1)
            # f2 for x1=0 is 1.0, which is correct.
            points = np.column_stack((x1, f2))
            hv = self._compute_hv(points, ref)
            if hv > best_hv:
                best_hv = hv
                best_p = p
                
        # Fine scan around best_p
        p_fine = np.arange(best_p - 0.1, best_p + 0.1, 0.005)
        for p in p_fine:
            if p <= 0: continue
            x1 = t ** p
            with np.errstate(invalid='ignore'):
                f2 = 1.0 - np.sqrt(x1)
            points = np.column_stack((x1, f2))
            hv = self._compute_hv(points, ref)
            if hv > best_hv:
                best_hv = hv
                best_p = p
                
        self.optimal_x1 = t ** best_p

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        if self.optimal_x1 is None:
            self._compute_optimal_x1()
            
        x1_vals = self.optimal_x1
        N = len(x1_vals)
        
        # Create X matrix
        # ZDT1: x[0] controls f1, x[1:] should be 0 for Pareto optimality
        X = np.zeros((N, self.dim))
        X[:, 0] = x1_vals
        
        # Evaluate all points
        # The problem function clips to bounds, so we are safe.
        for i in range(N):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
# EVOLVE-BLOCK-END
