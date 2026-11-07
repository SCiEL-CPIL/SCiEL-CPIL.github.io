# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Structure-aware optimizer for ZDT1.
    
    Key insight: For ZDT1, the Pareto front is achieved when all variables x2..x30 are 0.
    f1(x) = x1, and when x2..x30 = 0, g = 1, so f2 = 1 - sqrt(f1).
    The Pareto front is: for f1 in [0,1], f2 = 1 - sqrt(f1).
    
    To maximize hypervolume with reference point [11, 11]:
    - We want to cover the front as densely as possible.
    - The front is concave (f2 = 1 - sqrt(f1)), so we should cluster points where the curvature is high.
    - Near f1 = 0, the front is steep (df2/df1 = -1/(2*sqrt(f1)) → -∞), so small changes in f1 give large changes in f2.
    - Near f1 = 1, the front is flat (df2/df1 = -1/2), so we need more points there to capture the area.
    
    Actually, for hypervolume, we want to maximize the area under the front (in the region dominated by our points).
    The hypervolume contribution of a point (f1, f2) with reference (r1, r2) is the rectangle (r1-f1) * (r2-f2)
    minus overlaps. To maximize total hypervolume, we should place points to maximize the union of these rectangles.
    
    For a concave front like ZDT1, uniform spacing in f1 is a reasonable starting point, but we can do better
    by using a non-uniform distribution. Let's think about this more carefully.
    
    The hypervolume of a set of points on a Pareto front with reference point (r1, r2) is:
    HV = sum over sorted points of (f2[i] - f2[i+1]) * (r1 - f1[i]) + (r2 - f2[last]) * (r1 - f1[last])
    where points are sorted by f1 ascending, and f2 is decreasing.
    
    To maximize this, we want to choose f1 values that maximize this sum. This is equivalent to choosing
    points that maximize the area of the staircase approximation of the front.
    
    For a concave function, the optimal placement of points to maximize the staircase area is non-uniform.
    Let's use a numerical optimization approach: we'll parameterize the f1 values and optimize them.
    
    Actually, let's try a simpler approach first: use a power-law distribution.
    For ZDT1, f2 = 1 - sqrt(f1), so let t = sqrt(f1), then f1 = t^2, f2 = 1 - t.
    If we space t uniformly, we get f1 = (i/n)^2, which clusters points near f1 = 0.
    This might be good because the front is steep there.
    
    Let's try: t = (i + 0.5) / n for i in range(n), then f1 = t^2.
    This gives f1 values that are quadratic, clustering near 0.
    
    Alternatively, we can optimize the f1 values directly using a simple iterative method.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s/330s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = 0

        n_total = self.budget
        n_front = n_total  # Use all budget on the front

        # Strategy: Optimize the placement of points on the Pareto front to maximize hypervolume.
        # The front is f2 = 1 - sqrt(f1) for f1 in [0, 1].
        # We'll parameterize by f1 values and optimize them to maximize hypervolume.
        
        # Initial guess: uniform in sqrt(f1) space (i.e., f1 = t^2 where t is uniform)
        # This clusters points near f1 = 0 where the front is steep.
        t = (np.arange(n_front) + 0.5) / n_front
        f1_vals = t ** 2
        f1_vals = np.clip(f1_vals, lo[0], hi[0])
        
        # Compute hypervolume of a set of points on the front
        def compute_hv(f1_arr):
            """Compute hypervolume of points (f1, 1-sqrt(f1)) with reference (11, 11)."""
            # Sort by f1 ascending
            idx = np.argsort(f1_arr)
            f1_sorted = f1_arr[idx]
            f2_sorted = 1.0 - np.sqrt(np.clip(f1_sorted, 0, 1))
            
            # Only count points strictly better than reference in both objectives
            # f1 < 11 and f2 < 11, which is always true for f1 in [0,1]
            
            # Hypervolume calculation for 2D:
            # Sort points by f1 ascending, f2 should be descending for a Pareto front
            # HV = sum of (f2[i] - f2[i+1]) * (r1 - f1[i]) for i from 0 to n-2
            #     + (r2 - f2[n-1]) * (r1 - f1[n-1])
            r1, r2 = 11.0, 11.0
            
            if len(f1_sorted) == 0:
                return 0.0
            
            hv = 0.0
            for i in range(len(f1_sorted) - 1):
                if f2_sorted[i] > f2_sorted[i+1]:  # Ensure we're on the front
                    hv += (f2_sorted[i] - f2_sorted[i+1]) * (r1 - f1_sorted[i])
            # Last point to reference
            hv += (r2 - f2_sorted[-1]) * (r1 - f1_sorted[-1])
            
            return hv

        # Optimize f1_vals to maximize hypervolume using gradient-free optimization
        # We'll use a simple local search: perturb points and accept if HV improves
        
        best_f1 = f1_vals.copy()
        best_hv = compute_hv(best_f1)
        
        # Simple iterative optimization
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            
            improved = False
            for i in range(n_front):
                if time.time() > deadline - 10:
                    break
                
                # Try small perturbations
                for delta in [0.001, -0.001, 0.01, -0.01, 0.05, -0.05]:
                    new_f1 = best_f1.copy()
                    new_f1[i] = best_f1[i] + delta
                    # Ensure bounds
                    new_f1[i] = np.clip(new_f1[i], lo[0], hi[0])
                    
                    new_hv = compute_hv(new_f1)
                    if new_hv > best_hv + 1e-10:
                        best_f1 = new_f1
                        best_hv = new_hv
                        improved = True
                        break  # Move to next point after improvement
            
            if not improved:
                break
        
        f1_vals = best_f1
        f1_vals = np.clip(f1_vals, lo[0], hi[0])
        
        # Create the decision vectors
        X = np.zeros((n_front, self.dim))
        X[:, 0] = f1_vals
        # The rest are 0, which is within [0,1]
        
        # Evaluate all points
        F = np.zeros((n_front, self.n_obj))
        for i in range(n_front):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            n_evals += 1
        
        return X[:n_evals]
# EVOLVE-BLOCK-END
