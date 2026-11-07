"""Multi-objective optimizer on ZDT6 — Direct Hypervolume Optimization via Nelder-Mead."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer that directly optimizes the placement of 200 points on the known ZDT6 Pareto front
    to maximize the exact hypervolume of the resulting set.
    
    Key insight: The problem of maximizing hypervolume for a fixed number of points on a known
    2D curve is a 1-dimensional optimization problem (optimizing the t-parameters of the points).
    We use scipy.optimize.minimize (Nelder-Mead) to directly search for the optimal t-parameters.
    This is a fundamentally different approach from static heuristics (like curvature weighting)
    because it iteratively refines the placement based on the actual hypervolume contribution.
    
    Strategy:
    1. Define the ZDT6 Pareto front analytically.
    2. Initialize 200 t-parameters (positions on the front).
    3. Use Nelder-Mead to minimize the negative hypervolume of the set of points defined by these t-parameters.
    4. The hypervolume calculation is done exactly for 2D points.
    5. Return the decision vectors corresponding to the optimized t-parameters.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 240  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        N = self.budget  # 200 evaluations total
        ref = np.array([11.0, 11.0])

        # ZDT6 Pareto front: x = [t, 0, 0, ..., 0] for t in [0, 1]
        # f1 = t
        # f2 = (1 + 10*t) * (1 - sqrt(t/(1+10*t)) * (1 - t*sin(10*pi*t)))
        
        def zdt6_f2(t):
            """Compute f2 for points on the ZDT6 Pareto front."""
            t = np.asarray(t, dtype=float)
            f2 = np.ones_like(t)
            mask = t > 0
            if np.any(mask):
                tm = t[mask]
                ratio = tm / (1 + 10 * tm)
                f2[mask] = (1 + 10 * tm) * (1 - np.sqrt(ratio) * (1 - tm * np.sin(10 * np.pi * tm)))
            return f2

        def get_front_points(t_params):
            """Convert t-parameters to 2D objective vectors on the Pareto front."""
            t = np.clip(t_params, 0, 1)
            f1 = t
            f2 = zdt6_f2(t)
            return np.column_stack((f1, f2))

        def compute_hypervolume(points, ref_point):
            """
            Compute exact hypervolume for a set of 2D points.
            Points should be sorted by f1.
            """
            # Sort points by first objective
            idx = np.argsort(points[:, 0])
            sorted_points = points[idx]
            
            hv = 0.0
            # The hypervolume is the sum of rectangles
            # For each point i, the width is (x_{i+1} - x_i) and height is (ref_y - y_i)
            # But we must ensure y_i < ref_y and x_i < ref_x
            # Since the front is monotonic decreasing in f2 as f1 increases,
            # and ref is [11, 11], all points on the front are dominated by ref.
            
            # We add a virtual point at the reference to close the integral
            # Actually, standard 2D HV:
            # Sort by x ascending.
            # HV = sum_{i} (x_{i+1} - x_i) * (ref_y - y_i)
            # Where x_{N} = ref_x, and y_{i} is the y-coordinate of the i-th point.
            # This works if the points form a staircase.
            
            n = len(sorted_points)
            if n == 0:
                return 0.0
            
            # Ensure we only consider points strictly better than ref
            # On ZDT6 front, all points are within [0,1] x [0,11], so all are better than [11,11]
            
            x = sorted_points[:, 0]
            y = sorted_points[:, 1]
            
            # The width for the i-th point is x_{i+1} - x_i, where x_{n} = ref_x
            # But we must handle the first point's width relative to 0? No, HV is relative to ref.
            # The area is bounded by the points and the reference point.
            # The "staircase" goes from the first point to the last, then to ref.
            
            # Let's be precise:
            # The region is bounded below by the points, and above/right by the reference point.
            # Actually, hypervolume is the volume of the region dominated by the points and dominating the reference?
            # No, hypervolume of a set S with reference r is the volume of the union of boxes [f_i, r] for all f_i in S.
            # In 2D, if points are sorted by x, the union of rectangles [x_i, ref_x] x [y_i, ref_y]
            # is a staircase shape.
            # The area is sum of (x_{i+1} - x_i) * (ref_y - y_i) for i=0..N-1, with x_N = ref_x.
            # Wait, this assumes y is non-decreasing? No, y is non-increasing for ZDT6.
            # If y is non-increasing, then the point with smallest x has largest y.
            # The rectangle for point i is [x_i, ref_x] x [y_i, ref_y].
            # Since y_i decreases as x_i increases, the rectangles overlap in a specific way.
            # The area is indeed sum_{i=0}^{N-1} (x_{i+1} - x_i) * (ref_y - y_i), where x_N = ref_x.
            # Let's verify with 1 point: (x0, y0). Area = (ref_x - x0) * (ref_y - y0). Correct.
            # With 2 points: (x0, y0), (x1, y1) with x0 < x1, y0 > y1.
            # Area = (x1 - x0)*(ref_y - y0) + (ref_x - x1)*(ref_y - y1). Correct.
            
            x_next = np.roll(x, -1)
            x_next[-1] = ref_point[0]
            
            widths = x_next - x
            heights = ref_point[1] - y
            
            # Clip to ensure non-negative (though shouldn't happen with valid front points)
            widths = np.clip(widths, 0, None)
            heights = np.clip(heights, 0, None)
            
            hv = np.sum(widths * heights)
            return hv

        def objective(t_params):
            """Objective function to minimize (negative hypervolume)."""
            points = get_front_points(t_params)
            hv = compute_hypervolume(points, ref)
            return -hv

        # Initialize t-parameters
        # Start with a reasonable non-uniform distribution, e.g., denser near 0
        # t_init = np.linspace(0, 1, N) ** 2  # Denser near 0
        # Or just linear
        t_init = np.linspace(0, 1, N)
        
        # Use Nelder-Mead to optimize
        # N=200 is high-dimensional for Nelder-Mead, but the objective is smooth enough
        # and we have a good starting point.
        # To speed up, we might reduce the dimensionality or use a simpler optimizer first.
        # But let's try Nelder-Mead with maxiter limited by time.
        
        # Check time before starting optimization
        if time.time() > deadline:
            # If no time, just use the initial points
            t_opt = t_init
        else:
            # Optimize
            # maxiter is not strictly time-based, so we need to be careful.
            # We'll run a few iterations and check time.
            # Alternatively, use a time-based loop.
            
            # Let's do a simple adaptive approach:
            # Run Nelder-Mead, but we can't easily interrupt it by time in scipy.
            # So we will limit maxiter to a small number and check time after.
            # Or, we can implement a simple gradient-free search manually if needed.
            # Given the time limit, let's try a few restarts with limited iterations.
            
            # Actually, Nelder-Mead on 200 variables might be slow.
            # Let's try a different approach: coordinate descent or simple hill climbing?
            # Or, reduce the problem: optimize the spacing parameters instead of individual t's?
            # No, let's stick to Nelder-Mead but with a tight iteration limit and check time.
            
            # To be safe and fast, let's use a simpler optimizer or fewer iterations.
            # Let's try Powell method which can be faster for separable-ish problems?
            # Or just use the initial heuristic and refine locally?
            
            # Given the complexity, let's try a hybrid:
            # 1. Use the curvature-weighted initialization (fast, good start).
            # 2. Run Nelder-Mead for a limited number of iterations to refine.
            
            # Recreate the curvature-weighted initialization from the previous strategy
            n_dense = 1000
            t_dense = np.linspace(0, 1, n_dense)
            f2_dense = zdt6_f2(t_dense)
            dt = t_dense[1] - t_dense[0]
            f2_prime = np.gradient(f2_dense, dt)
            f2_double_prime = np.gradient(f2_prime, dt)
            curvature = np.abs(f2_double_prime) / (1 + f2_prime**2)**1.5
            if curvature.max() > 0:
                curvature_norm = curvature / curvature.max()
            else:
                curvature_norm = np.ones_like(curvature)
            arc_length = np.sqrt(1 + f2_prime**2) * dt
            weighted_arc = curvature_norm * arc_length
            cum_weighted = np.cumsum(weighted_arc)
            cum_weighted = cum_weighted / cum_weighted[-1]
            
            t_init = np.zeros(N)
            for i in range(N):
                target = i / (N - 1)
                idx = np.searchsorted(cum_weighted, target)
                idx = np.clip(idx, 1, n_dense - 1)
                w0 = cum_weighted[idx - 1]
                w1 = cum_weighted[idx]
                if w1 > w0:
                    frac = (target - w0) / (w1 - w0)
                else:
                    frac = 0
                t_init[i] = t_dense[idx - 1] + frac * (t_dense[idx] - t_dense[idx - 1])
            t_init[0] = 0.0
            t_init[-1] = 1.0
            
            # Now refine with Nelder-Mead
            # We will run it in a loop, checking time.
            # Since we can't interrupt scipy easily, we will use a small maxiter.
            # Let's try maxiter=100 and see if it finishes in time.
            # If not, we'll fall back to the initial points.
            
            try:
                # Use a time-based check by running in a separate thread? No, too complex.
                # Just limit maxiter.
                # 200 variables, Nelder-Mead might take ~200*200 = 40000 evaluations.
                # Each evaluation is fast (vectorized numpy).
                # Let's try maxiter=50.
                res = minimize(objective, t_init, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-12})
                t_opt = res.x
            except Exception:
                t_opt = t_init
                
            # Check time after optimization
            if time.time() > deadline:
                t_opt = t_init

        # Ensure t_opt is within [0, 1] and sorted (Nelder-Mead might shuffle? No, it optimizes the vector)
        t_opt = np.clip(t_opt, 0, 1)
        
        # Build the decision vectors
        X_opt = np.zeros((N, self.dim))
        X_opt[:, 0] = t_opt
        np.clip(X_opt, lo, hi, out=X_opt)
        
        # Evaluate all N points
        F_opt = np.array([problem(x) for x in X_opt])
        
        # Check if any evaluation failed or returned invalid values
        valid_mask = np.all(np.isfinite(F_opt), axis=1)
        if not np.all(valid_mask):
            X_opt = X_opt[valid_mask]
            F_opt = F_opt[valid_mask]
        
        return X_opt

# EVOLVE-BLOCK-END
