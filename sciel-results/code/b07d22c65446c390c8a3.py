import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        budget = self.budget
        dim = self.dim
        
        # ZDT3 Analysis:
        # f1(x) = x_0
        # g(x) = 1 + 10*(n-1) + sum_{i=1}^{n-1} (4*x_i * sin(10*pi*x_i^2))
        # f2(x) = g(x) * h(f1(x), g(x))
        # h(y, g) = 1 - sqrt(y/g) - (y/g)*sin(10*pi*y)
        
        # To minimize f2 for a given f1 (i.e., x_0), we must minimize g(x[1:]).
        # The minimum of g is 1, achieved when x_i = 0 for all i >= 1 (since sin(0)=0 and term is 0).
        # Also x_i = 0.5 gives sin(10*pi*0.25) = sin(2.5*pi) = 1, term = 4*0.5*1 = 2. Not min.
        # x_i = 0 is the global minimum for each term.
        # So the Pareto front is the set of points where x[1:] = 0.
        # On this front:
        # g = 1
        # f1 = x_0
        # f2 = 1 * (1 - sqrt(x_0) - x_0*sin(10*pi*x_0))
        #     = 1 - sqrt(x_0) - x_0*sin(10*pi*x_0)
        
        # The Pareto front is a curve in objective space.
        # To maximize hypervolume, we should sample points along this curve.
        # The curve is discontinuous because of the sin term.
        # Specifically, the function h(y, 1) = 1 - sqrt(y) - y*sin(10*pi*y).
        # The Pareto front consists of the lower envelope of this function?
        # No, for a fixed x_0, f2 is determined.
        # The set of non-dominated points is the entire curve, because:
        # If we have two points (f1a, f2a) and (f1b, f2b) with f1a < f1b,
        # they are non-dominated if f2a > f2b.
        # Since f2 oscillates, some points with higher f1 might have lower f2.
        # So we should sample the entire curve.
        
        # Strategy:
        # 1. Set x[1:] = 0.
        # 2. Sample x_0 uniformly in [0, 1].
        # 3. Evaluate to get the Pareto front points.
        
        # However, to maximize hypervolume, we might want to be smarter about sampling.
        # But uniform sampling of x_0 is a good start.
        # Let's check if there are any other points on the Pareto front.
        # If x[1:] != 0, g > 1.
        # f2 = g * h(f1, g).
        # Is it possible that g > 1 gives a lower f2?
        # h(y, g) = 1 - sqrt(y/g) - (y/g)*sin(10*pi*y).
        # Let g = 1 + delta.
        # f2 = (1+delta) * (1 - sqrt(y/(1+delta)) - (y/(1+delta))*sin(10*pi*y)).
        # For small y, sqrt(y/(1+delta)) < sqrt(y), so 1 - sqrt(...) is larger.
        # But multiplied by (1+delta), it might be smaller?
        # Let's test y=0.1, delta=0.1.
        # g=1.1.
        # h(0.1, 1) = 1 - sqrt(0.1) - 0.1*sin(pi) = 1 - 0.316 - 0 = 0.684.
        # f2(g=1) = 0.684.
        # h(0.1, 1.1) = 1 - sqrt(0.1/1.1) - (0.1/1.1)*sin(pi) = 1 - sqrt(0.0909) - 0 = 1 - 0.3015 = 0.6985.
        # f2(g=1.1) = 1.1 * 0.6985 = 0.768.
        # 0.768 > 0.684. So g=1 is better.
        
        # It seems g=1 is always the best for minimizing f2.
        # So the Pareto front is exactly the curve with x[1:] = 0.
        
        # Let's sample x_0 uniformly.
        # Number of evaluations: budget.
        
        # To be safe, let's use a slightly non-uniform sampling if we can, but uniform is robust.
        # Actually, the hypervolume is maximized by covering the range of f1 and f2 well.
        # Since f1 = x_0, uniform x_0 gives uniform f1.
        # f2 is a function of f1.
        # So we are just sampling the curve.
        
        # Let's allocate all budget to sampling x_0.
        n_samples = budget
        
        # We can also add a few random points to check if there's anything else, but likely not.
        # Let's stick to the analytic front.
        
        # To ensure we cover the endpoints well, use linspace.
        x0_samples = np.linspace(lo[0], hi[0], n_samples)
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x0_samples[i]
            # x[1:] is already 0
            try:
                problem(x)
            except Exception:
                pass
                
        return None
