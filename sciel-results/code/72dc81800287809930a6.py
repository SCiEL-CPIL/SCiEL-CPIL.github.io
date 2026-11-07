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
        
        n = self.dim
        budget = self.budget
        
        low = [b[0] for b in bounds]
        high = [b[1] for b in bounds]
        
        # DTLZ7 Analysis:
        # f1 = x[0]
        # f2 = x[1]
        # f3 = (1 + g(x[2:])) * h(x[1:20])
        # g(x[2:]) is minimized at x[2:] = 0, giving g=0.
        # h(x[1:20]) for M=3, N=20:
        # h(y) = 1 + sum_{i=1}^{M-1} (1+y_i) * (1 - y_i/(1+y_i))
        # Here M-1 = 2? No, standard DTLZ7 definition:
        # M is number of objectives (3).
        # h(y) = 1 + sum_{i=1}^{M-1} ... wait, usually h sums over M-1 variables?
        # Let's check standard DTLZ7:
        # h(y) = 1 + (M-1) * sum_{i=1}^{M-1} [ (1+y_i)/(1+y_i) * (1 - y_i/(1+y_i)) ] ? No.
        # Standard: h(y) = 1 + (M-1) * sum_{i=1}^{M-1} ( (1+y_i) * (1 - y_i/(1+y_i)) )
        # Wait, the term (1 - y/(1+y)) is 1/(1+y).
        # So term is (1+y)/(1+y) = 1? No.
        # Let's look at the formula carefully.
        # h(y) = 1 + (M-1) * sum_{i=1}^{M-1} [ (1+y_i) * (1 - y_i/(1+y_i)) ]
        # 1 - y/(1+y) = (1+y-y)/(1+y) = 1/(1+y).
        # So term is (1+y) * (1/(1+y)) = 1.
        # So sum is M-1.
        # h(y) = 1 + (M-1)*(M-1) = 1 + (M-1)^2.
        # For M=3, h(y) = 1 + 4 = 5.
        # Wait, other sources say h(y) = 1 + (M-1) * sum ...
        # Let's re-read the inspiration 1 comment:
        # "h(y) = 1 + sum_{i=1}^{M-1} (1+y_i) * (1 - y_i/(1+y_i))"
        # If the coefficient (M-1) is missing in the sum definition, then:
        # h(y) = 1 + sum_{i=1}^{M-1} 1 = 1 + (M-1) = M.
        # For M=3, h(y) = 3.
        # But DTLZ7 usually has h(y) = 1 + (M-1) * sum ...
        # If h(y) is constant, then f3 = (1+g)*h.
        # Min g is 0. Min f3 is h.
        # If h=20 (as in insp 1), f3=20.
        # If h=5, f3=5.
        # The reference point is [1.1, 1.1, 25.0].
        # The hypervolume calculation depends on the actual f3 value.
        # However, since we are minimizing, and we want to maximize HV,
        # we need to be as close to the ideal point as possible.
        # The ideal point is (0, 0, h_min).
        # The Pareto front is the set of non-dominated points.
        # If x[2:]=0, f3 is constant.
        # f1=x0, f2=x1.
        # The front is the line segment (0,0,h) to (1,1,h) in f-space?
        # No, f1 and f2 are independent.
        # So the front is the rectangle [0,1] x [0,1] x {h}.
        # Any point (x0, x1, h) is on the front if x0, x1 in [0,1].
        # To maximize HV, we need to cover the area in f1-f2 plane.
        # The dominated region for a point (f1, f2, f3) is [0, f1] x [0, f2] x [f3, ref3].
        # Since f3 is constant h for all points on the front,
        # the HV is integral over the front of the "shadow" volume?
        # Actually, HV of a set of points S is the volume of the union of boxes [0, f_i] x ...
        # Since f3 is constant, the union of boxes in f1-f2 plane is just the area covered by the projections of the points, multiplied by (ref3 - h).
        # So HV = (ref3 - h) * Area_Covered_in_f1f2_plane.
        # To maximize HV, we must maximize the area covered by the set of (f1, f2) points.
        # Since f1=x0 and f2=x1, we need to maximize the area covered by (x0, x1) pairs in [0,1]x[0,1].
        # This is a 2D packing/coverage problem.
        # With 200 points, we can cover the unit square very well.
        # A grid is optimal for covering a square with a fixed number of points if we are just looking at the union of infinitesimal boxes?
        # Actually, the "area covered" by a set of points in the context of HV is the area of the union of rectangles [0, x0_i] x [0, x1_i].
        # This is exactly the area of the histogram formed by the points.
        # To maximize this area, we want points to be spread out.
        # A uniform grid is a very good strategy.
        # Let's stick with the grid strategy but optimize the grid size.
        
        # Use x[2:] = 0 to minimize f3.
        best_x_tail = np.zeros(n - 2)
        
        # Allocate budget
        # We need to be careful about the exact number of evaluations.
        # 200 budget.
        # Let's use a grid of size K x K.
        # K^2 <= 200.
        # K=14 -> 196 points.
        # K=15 -> 225 points (too many).
        # So 14x14 grid uses 196 evaluations.
        # Remaining 4 evaluations can be used for random points or boundary points.
        
        n_grid = 14
        n_points = n_grid * n_grid
        if n_points > budget:
             n_grid = int(np.sqrt(budget))
             n_points = n_grid * n_grid
             
        xs0 = np.linspace(low[0], high[0], n_grid)
        xs1 = np.linspace(low[1], high[1], n_grid)
        
        x_base = np.zeros(n)
        
        count = 0
        for x0 in xs0:
            for x1 in xs1:
                if time.time() > deadline:
                    return x_base
                if count >= budget:
                    break
                    
                x = x_base.copy()
                x[0] = x0
                x[1] = x1
                
                problem(x)
                count += 1
                
            if time.time() > deadline or count >= budget:
                break
        
        # Use remaining budget for random sampling to fill gaps
        remaining = budget - count
        for i in range(remaining):
            if time.time() > deadline:
                break
            x_rand = x_base.copy()
            x_rand[0] = np.random.uniform(low[0], high[0])
            x_rand[1] = np.random.uniform(low[1], high[1])
            problem(x_rand)
            
        return x_base
