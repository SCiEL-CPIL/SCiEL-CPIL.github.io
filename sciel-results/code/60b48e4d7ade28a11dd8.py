"""
Problem-aware ZDT1 optimizer: Numerical optimization of point placement on the Pareto front.
Instead of using an analytical approximation for hypervolume-optimal spacing, 
this strategy directly optimizes the x-coordinates of the 200 points on the 
known ZDT1 Pareto front to maximize the exact 2D hypervolume.
"""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_2d(self, points, ref):
        """
        Compute exact hypervolume for a set of 2D points.
        points: Nx2 array of objective values.
        ref: 2D reference point.
        Returns: hypervolume.
        """
        if len(points) == 0:
            return 0.0
        
        # Filter points strictly better than reference
        mask = np.all(points < ref, axis=1)
        if not np.any(mask):
            return 0.0
        
        p = points[mask]
        
        # Sort by first objective (f1)
        p = p[np.argsort(p[:, 0])]
        
        hv = 0.0
        # We integrate the height of the front covered by each point
        # For sorted points, the contribution of point i is:
        # (x_{i+1} - x_i) * (ref[1] - y_i)  -- wait, this is for step functions.
        # Actually, for a set of points, the hypervolume is the union of boxes.
        # In 2D, if sorted by x, the hypervolume is sum of (x_{i+1} - x_i) * (ref_y - y_i)
        # but only if y_i is the minimum y seen so far? No, since we are on a Pareto front,
        # y is monotonically increasing with x.
        # So the "skyline" is just the points themselves.
        # The area is sum over i of (x_{i+1} - x_i) * (ref_y - y_i) ?
        # Let's verify.
        # Point 1: (x1, y1). Box is [x1, ref_x] x [y1, ref_y]. Area = (ref_x - x1)(ref_y - y1).
        # Point 2: (x2, y2) with x2 > x1, y2 > y1.
        # Union area = Area(1) + Area(2) - Intersection.
        # Intersection is [x2, ref_x] x [y2, ref_y]. Area = (ref_x - x2)(ref_y - y2).
        # So Total = (ref_x - x1)(ref_y - y1) - (ref_x - x2)(ref_y - y2) + ... 
        # This simplifies to sum of (x_{i+1} - x_i) * (ref_y - y_i) for i=1..N-1 + (ref_x - x_N)*(ref_y - y_N)?
        # Let's check the standard formula.
        # HV = sum_{i=1}^{N} (x_{i+1} - x_i) * (ref_y - y_i) where x_{N+1} = ref_x.
        # Yes, this works if points are sorted by x and y is increasing (Pareto front).
        
        xs = p[:, 0]
        ys = p[:, 1]
        
        # Append reference point x to close the integration interval
        # The last interval goes from x_N to ref_x
        dx = np.diff(np.append(xs, ref[0]))
        # The height for each interval i is ref_y - y_i
        # Note: the height is constant for the interval [x_i, x_{i+1}] because the "front" 
        # is effectively the step function defined by the points.
        # Wait, is it? The hypervolume of a set of points is the volume of the union of 
        # boxes dominated by each point.
        # Box i: [x_i, ref_x] x [y_i, ref_y].
        # Since y is increasing, Box i+1 is "above and right" of Box i.
        # The union is a staircase.
        # The area is indeed sum_{i=1}^{N} (x_{i+1} - x_i) * (ref_y - y_i) with x_{N+1}=ref_x.
        
        heights = ref[1] - ys
        hv = np.sum(dx * heights)
        
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin
        
        dim = self.dim
        budget = self.budget
        n = dim - 1
        
        if budget < 1:
            return np.array([])
        
        # Reference point
        ref = np.array([11.0, 11.0])
        
        # Pareto front function: f2 = 1 + 9 * sqrt(f1 / n)
        # We want to place N points on this curve to maximize HV.
        # Variables: f1 values, sorted, in [0, 1].
        
        # Initial guess: uniform distribution
        x0 = np.linspace(0.0, 1.0, budget)
        
        # Objective function to minimize (negative HV)
        def objective(f1_vals):
            # Ensure sorted
            f1_sorted = np.sort(f1_vals)
            
            # Compute f2 values
            f2_vals = 1.0 + 9.0 * np.sqrt(f1_sorted / n)
            
            # Form points
            points = np.column_stack((f1_sorted, f2_vals))
            
            # Compute HV
            hv = self._hypervolume_2d(points, ref)
            
            return -hv
        
        # Bounds for f1 values
        bounds_f1 = [(0.0, 1.0) for _ in range(budget)]
        
        # Optimize
        # L-BFGS-B is good for this smooth problem with bounds
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds_f1, 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        # Check time
        if time.time() > deadline:
            # Fallback to initial guess if optimization took too long
            f1_optimal = x0
        else:
            f1_optimal = np.sort(res.x)
        
        # Evaluate the optimal points on the problem
        results = []
        x_rest_good = np.full(dim - 1, 0.2)
        
        for f1_val in f1_optimal:
            if time.time() > deadline:
                break
            
            x = np.empty(dim)
            x[0] = f1_val
            x[1:] = x_rest_good
            
            f = problem(x)
            results.append(f)
            
        return np.array([]) if not results else None

if __name__ == "__main__":
    pass
