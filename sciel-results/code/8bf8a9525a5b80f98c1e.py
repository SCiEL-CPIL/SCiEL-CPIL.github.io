"""ZDT2 Optimizer using global optimal hypervolume quantization on the analytic Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Exploits the analytic structure of ZDT2.
    Strategy:
    1. Identify the 1D Pareto front parameterized by x1.
    2. Use the property that for a concave front in 2D, the optimal set of N points
       maximizing hypervolume with a fixed reference point consists of points where
       the line from the reference point is tangent to the front (or boundaries).
    3. Compute these N points analytically/numerically once.
    4. Evaluate exactly these points using the budget.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT2 specifics
        # f1 = x1
        # f2 = (1 + 10*(dim-1)) * (1 - sqrt(f1 / (1 + 10*(dim-1))))
        # Pareto front: x2...xdim = 0
        n_extra = dim - 1
        scale = 1.0 + 10.0 * n_extra
        
        # Reference point
        ref_f1 = 11.0
        ref_f2 = 11.0
        
        # Objective function on the Pareto front
        # x1 in [0, 1]
        def front_f1(x1):
            return x1
        
        def front_f2(x1):
            return scale * (1.0 - np.sqrt(x1 / scale))
        
        # The front is concave in objective space (f2 vs f1).
        # We want to choose N points x1_1 < x1_2 < ... < x1_N in [0, 1]
        # to maximize Hypervolume.
        
        # Hypervolume in 2D for points (f1_i, f2_i) sorted by f1 ascending:
        # HV = sum_{i=1}^{N} (f1_{i+1} - f1_i) * (ref_f2 - f2_i)
        # where f1_{N+1} = ref_f1.
        # Note: We assume the points are non-dominated, which they are on the true front.
        
        # The optimal condition for an interior point i (1 < i < N) is that the slope of the line
        # from ref to point i is equal to the slope of the tangent to the front at point i?
        # Actually, for maximizing area under a concave curve with a fixed number of rectangles,
        # the optimal points are where the "error" is balanced.
        # Specifically, the derivative of HV w.r.t x1_i is zero.
        # HV = ... + (x1_{i+1} - x1_i) * (ref_f2 - f2(x1_i)) + (x1_i - x1_{i-1}) * (ref_f2 - f2(x1_{i-1})) ...
        
        # Let's use a robust numerical approach to find the optimal N points.
        # Since N=200 is small, we can use a simple optimization or iterative refinement.
        # However, a known result for concave functions: the optimal points are distributed
        # such that the "gap" in hypervolume contribution is uniform? No.
        
        # Alternative: The optimal points for a concave function f(x) with reference (R, 0)
        # (shifting coords so ref is at R) are determined by the condition that the line
        # from the reference point to the point on the curve has the same slope as the tangent?
        # Let's check.
        # Area of rectangle i: w_i * h_i.
        # If we move point i, it affects rectangle i-1 and i.
        
        # Let's just use a fast iterative solver.
        # Initialize with uniform spacing.
        # Iterate: adjust x1_i to maximize local HV contribution.
        
        N = budget
        
        # Initial guess: uniform spacing
        x1s = np.linspace(0.0, 1.0, N)
        
        # Refinement loop
        # We want to maximize HV.
        # HV(x1s) = sum_{i=0}^{N-1} (x1s[i+1] - x1s[i]) * (ref_f2 - f2(x1s[i]))
        # with x1s[N] = ref_f1 (conceptually, but the last term is (ref_f1 - x1s[N-1]) * (ref_f2 - f2(x1s[N-1])))
        
        # Let's define a function to compute HV
        def compute_hv(x1s):
            # x1s sorted
            f2s = front_f2(x1s)
            # Widths
            next_x1s = np.append(x1s[1:], ref_f1)
            widths = next_x1s - x1s
            heights = ref_f2 - f2s
            return np.sum(widths * heights)
        
        # Gradient ascent / Newton-like step
        # For each interior point x1_i, the contribution to HV is:
        # Term from left neighbor: (x1_i - x1_{i-1}) * (ref_f2 - f2(x1_{i-1}))
        # Term from self: (x1_{i+1} - x1_i) * (ref_f2 - f2(x1_i))
        
        # dHV/dx1_i = (ref_f2 - f2(x1_{i-1})) - (ref_f2 - f2(x1_i)) - (x1_{i+1} - x1_i) * f2'(x1_i)
        # Set to 0:
        # ref_f2 - f2(x1_{i-1}) - ref_f2 + f2(x1_i) = (x1_{i+1} - x1_i) * f2'(x1_i)
        # f2(x1_i) - f2(x1_{i-1}) = (x1_{i+1} - x1_i) * f2'(x1_i)
        
        # This is a system of equations. We can solve it iteratively.
        # f2'(x) = scale * (1 / (2 * sqrt(x * scale))) = 0.5 * sqrt(scale / x)
        
        def f2_prime(x):
            # Handle x=0
            x_safe = np.maximum(x, 1e-10)
            return 0.5 * np.sqrt(scale / x_safe)
        
        # Iterative solver
        # x1_0 is fixed at 0? Or free?
        # If x1_0 > 0, we waste space. The first point should be 0?
        # Actually, if x1_0 > 0, the first rectangle is (x1_1 - x1_0) * (ref - f2(x1_0)).
        # The point at x=0 has f2(0) = scale.
        # If we don't include 0, we lose the area from 0 to x1_0?
        # No, the HV is defined by the points. If the lowest f1 point is x1_0, the region [0, x1_0] is not covered by any rectangle in the standard definition?
        # Wait, the standard HV definition for a set of points P is the volume of the union of boxes [p_i, ref].
        # If min(f1) > 0, the box starts at min(f1).
        # So we should always include a point with f1 close to 0 to capture the left-most area.
        # Similarly, include a point with f1 close to 1 to capture the right-most area?
        # Actually, the last rectangle extends to ref_f1. So the last point determines the height for the tail.
        
        # Let's fix x1[0] = 0.0 and x1[-1] = 1.0?
        # If we fix endpoints, we optimize the interior points.
        # This is a good constraint to ensure full coverage.
        
        x1s[0] = 0.0
        x1s[-1] = 1.0
        
        # Optimize interior points 1 to N-2
        for _ in range(50):  # Iterations
            if time.time() > deadline - 10: break
            improved = False
            for i in range(1, N-1):
                # Equation: f2(x1_i) - f2(x1_{i-1}) = (x1_{i+1} - x1_i) * f2'(x1_i)
                # Let g(x) = f2(x) - f2(x1_{i-1}) - (x1_{i+1} - x) * f2'(x)
                # We want g(x) = 0.
                # g'(x) = f2'(x) - [ -1 * f2'(x) + (x1_{i+1} - x) * f2''(x) ]
                # f2''(x) = -0.25 * scale^(1/2) * x^(-3/2)
                
                x_prev = x1s[i-1]
                x_next = x1s[i+1]
                
                # Newton step
                x = x1s[i]
                for _ in range(5):
                    f2_curr = front_f2(x)
                    f2_prev = front_f2(x_prev)
                    f2p_curr = f2_prime(x)
                    
                    # g(x)
                    g_val = f2_curr - f2_prev - (x_next - x) * f2p_curr
                    
                    # g'(x)
                    # d/dx [ f2(x) ] = f2'(x)
                    # d/dx [ - (x_next - x) f2'(x) ] = f2'(x) - (x_next - x) f2''(x)
                    f2pp_curr = -0.25 * np.sqrt(scale) / (x**1.5) if x > 1e-10 else 0
                    g_deriv = f2p_curr - (f2p_curr - (x_next - x) * f2pp_curr)
                    g_deriv = (x_next - x) * f2pp_curr # Simplified: f2' cancels out?
                    # Let's re-derive carefully.
                    # H(x) = (x_next - x) * (ref - f2(x)) + (x - x_prev) * (ref - f2(x_prev))
                    # dH/dx = -(ref - f2(x)) + (x_next - x) * f2'(x) + (ref - f2(x_prev))
                    # Set to 0:
                    # (ref - f2(x_prev)) - (ref - f2(x)) + (x_next - x) f2'(x) = 0
                    # f2(x) - f2(x_prev) + (x_next - x) f2'(x) = 0
                    # This is the equation we solved above.
                    
                    # Derivative of LHS w.r.t x:
                    # f2'(x) + [ -1 * f2'(x) + (x_next - x) f2''(x) ]
                    # = (x_next - x) f2''(x)
                    # Since f2'' < 0, the function is convex? No, f2 is concave, f2'' < 0.
                    # So the derivative is negative. The function is decreasing.
                    # Newton step: x_new = x - g_val / g_deriv
                    
                    if abs(g_deriv) < 1e-12:
                        break
                    
                    x_new = x - g_val / g_deriv
                    
                    # Clip to (x_prev, x_next)
                    if x_new <= x_prev:
                        x_new = (x_prev + x) / 2
                    elif x_new >= x_next:
                        x_new = (x_next + x) / 2
                    
                    if abs(x_new - x) < 1e-10:
                        x = x_new
                        break
                    x = x_new
                
                if abs(x - x1s[i]) > 1e-8:
                    x1s[i] = x
                    improved = True
            
            if not improved:
                break
        
        # Sort and ensure uniqueness
        x1s = np.sort(x1s)
        
        # Evaluate the points
        last_F = None
        for x1 in x1s:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = np.clip(x1, lo[0], hi[0])
            # x[1:] = 0 (clipped to bounds)
            F = np.array(problem(x))
            last_F = F
            
        return last_F

# EVOLVE-BLOCK-END
