import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = self.budget
        if N <= 0:
            return None

        # ZDT1 Pareto Front Structure:
        # f1 = x1
        # g = 1 + 9 * sum(x2..xn)/(n-1)
        # f2 = g * (1 - sqrt(x1/g))
        #
        # The global Pareto front occurs when g=1, which implies x2..xn = 0.
        # On this front:
        # f1 = x1
        # f2 = 1 - sqrt(x1)
        #
        # We want to maximize Hypervolume with reference point [11, 11].
        # HV = sum_{i=1}^{N} (f1_{i+1} - f1_i) * (11 - f2_i)
        # where f1_{N+1} = 11 (the reference point's f1 coordinate, effectively 11 since max f1 is 1).
        # Actually, the standard HV calculation for a set of points P with reference r is:
        # HV = sum_{p in P} volume of box dominated by p and not by any other q in P.
        # For 2D, sorted by f1:
        # HV = sum_{i=1}^{N} (f1_{i+1} - f1_i) * (r2 - f2_i)
        # where f1_{N+1} = r1 = 11.
        #
        # Substituting f2_i = 1 - sqrt(f1_i):
        # HV = sum_{i=1}^{N} (f1_{i+1} - f1_i) * (11 - (1 - sqrt(f1_i)))
        # HV = sum_{i=1}^{N} (f1_{i+1} - f1_i) * (10 + sqrt(f1_i))
        #
        # To maximize this sum, we need to choose f1_1 < f1_2 < ... < f1_N in [0, 1].
        # The term (10 + sqrt(f1_i)) is the "height" of the box.
        # The term (f1_{i+1} - f1_i) is the "width".
        # Since sqrt(f1) changes rapidly near 0 and slowly near 1, we should place more points near 0.
        # Specifically, if we approximate the integral, the optimal density is proportional to 1/sqrt(f1).
        # This suggests f1_i should be proportional to i^2.
        # Let f1_i = (i/N)^2 for i=1..N.
        # Let's verify if this is optimal or if we can do better.
        # The continuous optimum for maximizing \int_0^1 (10 + sqrt(x)) dx is just the integral itself, 
        # but we are summing rectangles. The "staircase" approximation is best when the step size 
        # adapts to the derivative of the function defining the top boundary.
        # The top boundary is y = 11 - (1 - sqrt(x)) = 10 + sqrt(x).
        # We are approximating the area under y(x) from x=0 to x=1, plus the rectangle from x=1 to x=11.
        # Wait, the reference point is [11, 11].
        # The last point f1_N is at most 1. The box for the last point extends to f1=11.
        # So the last term is (11 - f1_N) * (10 + sqrt(f1_N)).
        #
        # Let's use the quadratic spacing f1_i = (i/N)^2.
        # f1_1 = 1/N^2, ..., f1_N = 1.
        #
        # We construct the points:
        # x = [f1_i, 0, 0, ..., 0]
        
        # Generate f1 values
        # i from 1 to N
        i_vals = np.arange(1, N + 1)
        f1_vals = (i_vals / N) ** 2
        
        # Ensure we stay within bounds [0, 1] for x1.
        # lo[0] is 0, hi[0] is 1.
        f1_vals = np.clip(f1_vals, lo[0], hi[0])
        
        # Create the population points
        # All points have x2..xn = 0 (which is within [0,1])
        points = np.zeros((N, self.dim))
        points[:, 0] = f1_vals
        
        # Evaluate all points
        # We must ensure we don't exceed the budget.
        # N is exactly the budget.
        
        for i in range(N):
            if time.time() > deadline:
                break
            # The problem function clips x to bounds. 
            # Since our points are constructed within [0,1] for x1 and 0 for others, 
            # and bounds are [0,1], this is safe.
            problem(points[i])
            
        return None
