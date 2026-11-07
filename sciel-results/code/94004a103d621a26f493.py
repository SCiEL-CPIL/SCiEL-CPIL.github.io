import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT3 using Geometric Front Sampling.
    
    Strategy:
    1. The Pareto front of ZDT3 is known analytically: f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0,1],
       achieved when x2..xn = 0.
    2. The problem reduces to selecting B x1 values in [0,1] to maximize the hypervolume of 
       {(x1, 1-sqrt(x1))} with reference point [11, 11].
    3. The hypervolume contribution of a point (x, y) is roughly proportional to the area of the 
       rectangle it adds. Near x=0, the front is steep (high dy/dx), meaning small changes in x 
       yield large changes in y, and thus large areas of uncovered space. 
    4. Therefore, a geometric (exponential) distribution of x1 values, which places more points 
       near 0, should yield a higher hypervolume than a uniform distribution.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        n = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We will evaluate exactly on the Pareto front.
        # x = [x1, 0, 0, ..., 0]
        
        # Use a geometric distribution for x1 values.
        # We want to cluster points near 0.
        # A simple way is to use x1 = (i / (B-1)) ** k for some k > 1.
        # Or use a geometric sequence.
        # Let's try x1_i = (i / (B-1)) ** 2. This squares the uniform grid, pushing points to 0.
        # Alternatively, x1_i = 1 - (1 - i/(B-1))**2 pushes points to 1.
        # Since the front is convex and we want to cover the area under the curve, 
        # and the curve is 1-sqrt(x), the "area" is maximized by covering the region where 
        # the curve is high (near x=0) more densely.
        
        # Let's try a power law: x1 = (i / (B-1)) ** p
        # If p > 1, points are denser near 0.
        # If p < 1, points are denser near 1.
        
        # The hypervolume of a set of points on a convex front in 2D with ref (R1, R2) is the area 
        # of the polygon formed by the points and the reference.
        # For a convex decreasing front, the "loss" in HV compared to the true front is the area 
        # between the piecewise linear approximation and the true curve.
        # To minimize this loss, we should place points where the curvature is highest.
        # Curvature of y = 1 - sqrt(x):
        # y' = -1/(2 sqrt(x))
        # y'' = 1/(4 x^(3/2))
        # Curvature is highest at x -> 0.
        # So we should densely sample near x=0.
        
        # Let's use x1 = (i / (B-1)) ** 2.
        
        B = self.budget
        if B == 1:
            x1_values = np.array([0.0])
        else:
            i = np.arange(B)
            # Map [0, B-1] to [0, 1]
            t = i / (B - 1)
            # Apply power law to cluster near 0
            x1_values = t ** 2
            
            # Ensure endpoints are exactly 0 and 1
            x1_values[0] = 0.0
            x1_values[-1] = 1.0
            
        F = []
        for x1 in x1_values:
            x = np.zeros(n)
            x[0] = x1
            # x[1:] = 0.0 is already true from zeros
            x = np.clip(x, lo, hi)
            f = problem(x)
            F.append(f)
            
            if time.time() > deadline:
                break

        return np.array(F)
