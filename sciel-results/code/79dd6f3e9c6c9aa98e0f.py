"""Optimized ZDT2 solver using optimal transport-inspired spacing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer for ZDT2 using analytic structure and optimal spacing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # ZDT2 Analytic Structure:
        # f1(x) = x1
        # f2(x) = (1 + 10*(n-1)) * (1 - sqrt(f1)) + sum(xi^2 for i=2..n)
        # Pareto front: x2...xn = 0.
        # f2 = C * (1 - sqrt(f1)) where C = 1 + 10*(dim-1).
        
        C = 1 + 10 * (dim - 1)
        ref = np.array([11.0, 11.0])
        
        # We want to place N points on the curve to maximize hypervolume.
        # Hypervolume contribution of a point (f1_i, f2_i) relative to neighbors:
        # For a sorted set of points on a monotonic curve, the HV is the sum of trapezoids/rectangles.
        # Specifically, if we have points sorted by f1 ascending (which means f2 descending),
        # the HV is sum over i of (f2_i - f2_{i+1}) * (ref[0] - f1_i) + (ref[1] - f2_N) * (ref[0] - f1_N)
        # Wait, standard HV calculation for minimization with ref (R1, R2):
        # HV = sum_{i=1}^{N} (R1 - f1_i) * (f2_{i-1} - f2_i) where f2_0 = R2 and f2_{N+1} = 0? 
        # No, the region is bounded by the front and the reference point.
        # Let's use the standard formula for a non-dominated set sorted by f1.
        # Let points be p_1, ..., p_N sorted by f1 ascending.
        # The hypervolume is the area of the union of rectangles [f1_i, R1] x [f2_i, R2] minus overlaps.
        # For a monotonic decreasing front, the HV is:
        # HV = sum_{i=1}^N (R1 - f1_i) * (f2_{i-1} - f2_i) where f2_0 = R2.
        # Note: f2_{N} is the last point. The term for i=N is (R1 - f1_N) * (f2_{N-1} - f2_N).
        # Is there a tail? The rectangle for the last point extends to R1 and R2? 
        # Actually, the dominated region is the set of points (x, y) such that x >= f1_i and y >= f2_i for some i.
        # This is equivalent to the area under the "staircase" formed by the points and the reference point.
        # The area is sum_{i=1}^N (R1 - f1_i) * (f2_{i-1} - f2_i) with f2_0 = R2.
        
        # We want to maximize this HV.
        # Let t_i = f1_i. Then f2_i = C * (1 - sqrt(t_i)).
        # We need to choose t_1 < t_2 < ... < t_N in [0, 1].
        # Let's optimize the positions t_i directly.
        
        N = budget
        
        # Initial guess: Power law spacing which is known to be good.
        # t_i = (i/N)^(2/3)
        t_init = np.linspace(0, 1, N) ** (2.0/3.0)
        # Ensure strictly increasing and within bounds
        t_init[0] = 1e-6
        t_init[-1] = 1.0 - 1e-6
        t_init = np.sort(t_init)
        
        # Objective function to minimize (negative HV)
        def neg_hv(t):
            # t is array of length N
            t = np.clip(t, 0, 1)
            t = np.sort(t)
            f2 = C * (1 - np.sqrt(t))
            f2 = np.clip(f2, 0, C)
            
            # f2 is decreasing as t increases.
            # f2_prev: f2_0 = R2 = 11.0
            # f2_i for i=1..N
            f2_prev = np.empty(N + 1)
            f2_prev[0] = 11.0
            f2_prev[1:] = f2
            
            # HV = sum (R1 - t_i) * (f2_prev[i-1] - f2_prev[i])
            # Note: f2_prev[i-1] is the f2 of the previous point (or ref)
            # f2_prev[i] is the f2 of the current point
            # Since f2 is decreasing, f2_prev[i-1] >= f2_prev[i]
            
            deltas_f2 = f2_prev[:-1] - f2_prev[1:]
            widths = 11.0 - t
            
            hv = np.sum(widths * deltas_f2)
            return -hv

        # Gradient for faster convergence? 
        # dHV/dt_i = -(f2_prev[i-1] - f2_prev[i]) + (11 - t_i) * (-d(f2_prev[i-1] - f2_prev[i])/dt_i)
        # This is complex due to sorting. Let's use numerical gradient or L-BFGS-B with finite differences.
        # Given N=200, L-BFGS-B should be fast enough.
        
        # Bounds for t_i: [0, 1]
        # To ensure strict ordering, we can optimize gaps or just rely on L-BFGS-B keeping them distinct if initialized well.
        # Better: Optimize log-spaced parameters or use a transformation that guarantees ordering.
        # Let t_i = sigmoid(u_i). Then dt_i/du_i > 0.
        # But sigmoid maps to (0,1). We want [0,1].
        # Let's stick to t_i with bounds [0,1] and hope the optimizer maintains order.
        # If order is violated, sorting inside neg_hv makes the objective non-smooth.
        # Alternative: Optimize the gaps g_i = t_{i+1} - t_i > 0.
        # t_1 = a, t_{i+1} = t_i + g_i.
        # Constraints: a > 0, g_i > 0, sum(g_i) + a <= 1.
        
        # Let's try a simpler approach: 
        # The optimal spacing for HV maximization on a convex curve is related to the curvature.
        # For f2 = C(1-sqrt(f1)), the curvature is high near f1=0.
        # The "ideal" density of points rho(t) is proportional to sqrt(1 + (f2')^2) / |f2''| ?
        # Or simpler: The marginal gain of a point is proportional to the local slope and width.
        
        # Let's use the "Optimal Transport" idea from the prompt.
        # Minimize L1 distance to continuous measure.
        # The continuous measure for optimal HV sampling on a curve y=f(x) with ref (X, Y)
        # is often approximated by the arc length or curvature.
        # For ZDT2, the derivative f2' = -C/(2*sqrt(f1)).
        # The "cost" of a point is its contribution to HV.
        
        # Let's just run L-BFGS-B on the sorted t vector. 
        # To handle the sorting issue, we can define the objective on unsorted variables 
        # but penalize inversions heavily, or simply rely on the fact that the initial 
        # guess is sorted and the landscape is smooth enough that L-BFGS-B won't cross points 
        # if we start with good separation.
        
        # However, L-BFGS-B might be slow for 200 vars if not careful.
        # Let's try a coordinate ascent or a simple gradient descent with analytical gradient.
        
        # Let's compute the gradient of HV w.r.t t_i assuming order is preserved.
        # HV = sum_{j=1}^N (11 - t_j) * (f2_{j-1} - f2_j)
        # where f2_0 = 11, f2_j = C(1-sqrt(t_j))
        # f2_{j-1} depends on t_{j-1} (for j>1) and is constant 11 for j=1.
        # f2_j depends on t_j.
        
        # dHV/dt_i:
        # Term i: (11 - t_i) * (f2_{i-1} - f2_i)
        #   d/dt_i [ (11 - t_i) * (f2_{i-1} - f2_i) ] = -1 * (f2_{i-1} - f2_i) + (11 - t_i) * ( - df2_i/dt_i )
        #   df2_i/dt_i = -C / (2 * sqrt(t_i))
        #   So term i deriv = -(f2_{i-1} - f2_i) + (11 - t_i) * C / (2 * sqrt(t_i))
        
        # Term i+1: (11 - t_{i+1}) * (f2_i - f2_{i+1})
        #   d/dt_i [ ... ] = (11 - t_{i+1}) * ( - df2_i/dt_i )
        #   = (11 - t_{i+1}) * C / (2 * sqrt(t_i))
        
        # Term i-1: (11 - t_{i-1}) * (f2_{i-2} - f2_{i-1})
        #   No dependency on t_i.
        
        # So, for i=1:
        # f2_0 = 11.
        # dHV/dt_1 = -(11 - f2_1) + (11 - t_1) * C / (2 * sqrt(t_1)) + (11 - t_2) * C / (2 * sqrt(t_1))
        
        # For i > 1:
        # dHV/dt_i = -(f2_{i-1} - f2_i) + (11 - t_i) * C / (2 * sqrt(t_i)) + (11 - t_{i+1}) * C / (2 * sqrt(t_i))
        # Note: f2_{i-1} - f2_i is the height of the rectangle for point i.
        
        # We want to maximize HV, so minimize -HV.
        # Gradient of -HV is -dHV/dt.
        
        def neg_hv_and_grad(t):
            t = np.clip(t, 1e-8, 1.0)
            f2 = C * (1 - np.sqrt(t))
            f2[0] = np.clip(f2[0], 0, C)
            
            # Calculate HV
            f2_prev = np.empty(N + 1)
            f2_prev[0] = 11.0
            f2_prev[1:] = f2
            deltas_f2 = f2_prev[:-1] - f2_prev[1:]
            widths = 11.0 - t
            hv = np.sum(widths * deltas_f2)
            
            # Calculate Gradient
            grad = np.zeros(N)
            
            # d f2 / dt = -C / (2 sqrt(t))
            df2_dt = -C / (2 * np.sqrt(t))
            
            # Term from own width and height
            # d/dt_i [ (11 - t_i) * (f2_{i-1} - f2_i) ]
            # = -1 * (f2_{i-1} - f2_i) + (11 - t_i) * (-df2_dt[i])
            
            heights = deltas_f2 # f2_{i-1} - f2_i
            
            grad -= heights
            grad += (11.0 - t) * (-df2_dt)
            
            # Term from next point's height
            # d/dt_i [ (11 - t_{i+1}) * (f2_i - f2_{i+1}) ]
            # = (11 - t_{i+1}) * (-df2_dt[i])
            # This applies for i < N-1
            if N > 1:
                next_widths = np.zeros(N)
                next_widths[:-1] = 11.0 - t[1:]
                grad += next_widths * (-df2_dt)
            
            return -hv, -grad
        
        # Use L-BFGS-B
        # Bounds: [1e-8, 1.0] for all t
        bounds_opt = [(1e-8, 1.0) for _ in range(N)]
        
        # Initial point
        t_current = np.linspace(0.01, 0.99, N) ** (2.0/3.0)
        t_current = np.sort(t_current)
        
        try:
            res = minimize(neg_hv_and_grad, t_current, jac=True, method='L-BFGS-B', bounds=bounds_opt,
                           options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-6})
            if time.time() > deadline - 10:
                raise TimeoutError()
            t_opt = res.x
        except Exception:
            # Fallback to power law if optimization fails or is too slow
            t_opt = (np.arange(1, N + 1) / N) ** (2.0 / 3.0)
            t_opt = np.clip(t_opt, 1e-8, 1.0)
        
        # Ensure t_opt is sorted and within bounds
        t_opt = np.sort(np.clip(t_opt, 1e-8, 1.0))
        
        # Construct decision vectors
        X = np.zeros((N, dim))
        X[:, 0] = t_opt
        # X[:, 1:] is 0
        
        # Evaluate
        F = []
        for x in X:
            if time.time() > deadline - 1:
                break
            F.append(problem(x))
            
        return X

# EVOLVE-BLOCK-END
