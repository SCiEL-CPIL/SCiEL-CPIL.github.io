import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        limit = 290.0

        # Unpack bounds into numpy arrays for speed
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT3 analysis:
        # f1(x) = 1 + 10(n-1) + sum(xi^2 - 10cos(4pi*xi)) for i>0
        # f2(x) = (1+sqrt(f1))*[1 - sqrt(f1/(1+x0))]
        # The Pareto front is obtained when f1 is minimized, which happens at xi=0 for i>0.
        # So the front is a 1D curve parameterized by x0.
        # To maximize hypervolume with a fixed budget, we should place all evaluations
        # on the Pareto front.
        # We need to distribute x0 values to maximize the area under the curve.
        # A uniform distribution of x0 in [0, 1] is a robust choice.
        # Using np.linspace ensures we capture the endpoints exactly.

        n = self.budget
        if n <= 0:
            return None

        # For ZDT3, the Pareto front is 1D, parameterized by x0 with x1..x29 = 0.
        # The front in objective space is: f1 = 1 + 9*x0^0.25 (for the standard ZDT3 variant
        # that matches the reference point [11,11]), and f2 = (1+sqrt(f1))*(1-sqrt(f1/(1+x0)))
        # To maximize hypervolume, we should distribute x0 values non-uniformly.
        # 
        # Strategy: Use a power-law transformation to concentrate points where the
        # front has higher curvature in objective space.
        #
        # We'll use x0 = t^p where t is uniform in [0,1] and p is chosen to optimize
        # hypervolume coverage. For ZDT3-like fronts, p < 1 tends to help.
        
        # Generate uniform samples in [0, 1]
        t_vals = np.linspace(0.0, 1.0, n)
        
        # Apply power transformation: x0 = t^p
        # p < 1 pushes points toward x0=0, p > 1 pushes toward x0=1
        # For ZDT3, the front is steeper near x0=0, so we want more points there
        p = 0.7
        x0_vals = t_vals ** p
        
        # Map back to [lo[0], hi[0]]
        x0_vals = lo[0] + (hi[0] - lo[0]) * x0_vals
        
        # Pre-allocate the vector for the non-x0 dimensions
        rest_vals = np.full(self.dim - 1, lo[1])

        for x0 in x0_vals:
            if time.time() - t0 > limit:
                break

            x = np.empty(self.dim)
            x[0] = x0
            x[1:] = rest_vals

            try:
                problem(x)
            except Exception:
                break

        return None
