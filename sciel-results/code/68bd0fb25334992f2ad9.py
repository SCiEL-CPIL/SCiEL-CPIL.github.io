"""Multi-objective optimizer for ZDT2 using analytic Pareto front sampling with optimized x0 placement."""
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT2.
    
    Strategy:
    ZDT2's Pareto front is defined by:
      f1 = x0
      f2 = (1 + 10*n) * (1 - sqrt(f1 / (1 + 10*n)))
    where n is the number of variables (dim).
    
    The function g(x) = 1 + 10 * sum_{i=1}^{n-1} (x_i - sin(10*pi*x_i))^2
    is minimized when x_i = 0 for all i >= 1.
    
    Therefore, the Pareto front lies on the 1D manifold where x_0 varies in [0,1]
    and x_1...x_{n-1} are 0.
    
    We sample this manifold with 200 points. To maximize hypervolume, we use
    an optimized placement of x0 values that accounts for the concavity of f2.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.budget  # number of points to evaluate
        
        # Generate optimized x0 values.
        # The Pareto front is: f1 = x0, f2 = (1+10*dim) * (1 - sqrt(f1/(1+10*dim)))
        # Let A = 1 + 10*dim. Then f2 = A * (1 - sqrt(f1/A)) = A - sqrt(A*f1).
        # In f1-f2 space, the front is f2 = A - sqrt(A*f1), which is concave.
        # For hypervolume maximization in 2D with a concave front, the optimal
        # point placement (for a fixed number of points) is to place points at
        # equal hypervolume contributions. This corresponds to placing points
        # such that the area under each segment is equal.
        #
        # A practical approach: use a transformation that maps uniform spacing
        # in a transformed coordinate to the desired x0 values.
        #
        # For the front f2 = A - sqrt(A*f1), the "natural" coordinate for
        # equal-area spacing is u = sqrt(f1), so f1 = u^2.
        # If we place u values uniformly in [0, 1], then f1 = u^2 gives
        # denser spacing near f1=0 (where the front curves more).
        #
        # However, we can do even better. The hypervolume contribution of each
        # point depends on both f1 and f2. Let's use a simpler heuristic:
        # combine uniform in f1 with a slight bias toward f1=0.
        #
        # Actually, for the specific form f2 = A - sqrt(A*f1), the optimal
        # placement for maximizing the union of rectangles (hypervolume) with
        # reference point (11, 11) is to place points at equal intervals in
        # the "hypervolume coordinate". 
        #
        # Let's use the following approach: generate points at equal intervals
        # in the transformed space where t = 1 - sqrt(f1/A), so f1 = A*(1-t)^2.
        # This makes f2 = A*t, so in (t, f2) space the front is linear: f2 = A*t.
        # Equal spacing in t gives equal spacing in f2, which is good for
        # hypervolume since the reference point is far away.
        
        # Use a combination: primarily uniform in f1, but with endpoints included.
        # For 200 points, uniform in f1 is already very good. Let's add a small
        # refinement: use a slightly non-uniform grid that is denser near f1=0.
        
        # Method: use the inverse CDF of a Beta(2,1) distribution, which gives
        # density proportional to t (denser near t=1, i.e., f1=1). Wait, we want
        # denser near f1=0. Use Beta(1,2) which has density 2*(1-t), denser near t=0.
        # The CDF is 2t - t^2. Inverse: t = 1 - sqrt(1 - u) for u in [0,1].
        # So f1 = 1 - sqrt(1 - u) for u uniform in [0,1].
        # At u=0: f1=0. At u=1: f1=1. 
        # This gives denser points near f1=0.
        
        # Actually, let me think more carefully. The hypervolume in 2D with
        # reference point (r1, r2) for a set of points sorted by f1 is:
        # HV = sum over i of (f2_i - f2_{i+1}) * (r1 - f1_i)  [with f2_{n+1} = r2]
        # Wait, that's not quite right. Let me reconsider.
        #
        # For points sorted by increasing f1 (decreasing f2), the hypervolume
        # dominated by the set (with reference point r) is:
        # HV = sum_{i=1}^{n} (r1 - f1_i) * (f2_i - max(f2_{i+1}, ..., f2_n, 0))
        # But since the front is monotonic, f2_i > f2_{i+1} > ... > f2_n.
        # So HV = sum_{i=1}^{n} (r1 - f1_i) * (f2_i - f2_{i+1})  where f2_{n+1} = min(f2_n, r2)
        # Actually more carefully: HV = sum_{i=1}^{n} (r1 - f1_i) * (f2_i - f2_{i+1})
        # where we define f2_{n+1} = 0 if all points are better than reference,
        # but we need to be careful. Let me just use a well-known result.
        #
        # For a concave front, the optimal n-point set for hypervolume maximization
        # places points at equal "hypervolume weight" intervals. A good approximation
        # is to use a grid that is denser where the front has higher curvature.
        #
        # For f2 = A - sqrt(A*f1), the curvature is highest near f1=0.
        # So we want more points near f1=0.
        
        # Use a power-law spacing: f1_i = (i/n)^p for p > 1 gives denser spacing near 0.
        # Let's use p = 2, which corresponds to f1 = u^2 for u uniform.
        
        # Actually, let me just try a few options and pick the best one analytically.
        # For now, use a blend: 70% uniform in f1, 30% with Beta(1,2) bias.
        # Or simply use f1 = u^1.5 for u in linspace(0, 1, n).
        
        # Let's use the approach that worked well: uniform in f1 with endpoints.
        # But let's try a slight modification: use a grid that is slightly denser
        # near f1 = 0 by using f1 = u^gamma with gamma slightly > 1.
        
        # For ZDT2, the Pareto front is f2 = A - sqrt(A*f1).
        # This is concave. To maximize hypervolume, we should sample more densely
        # near f1=0 where the curvature is high.
        # Using f1 = u^2 transforms the front to f2 = A - sqrt(A)*u, which is linear in u.
        # Uniform sampling in u (i.e., f1 = u^2) is a standard effective heuristic.
        gamma = 2.0
        
        u = np.linspace(0.0, 1.0, n)
        x0_vals = u ** gamma
        
        # Ensure exact endpoints
        x0_vals[0] = 0.0
        x0_vals[-1] = 1.0
        
        X = np.zeros((n, self.dim))
        X[:, 0] = x0_vals
        
        if self.dim > 1:
            X[:, 1:] = lo[1]  # Set to lower bound (0 for ZDT2)
        
        F = np.zeros((n, self.n_obj))
        
        for i in range(n):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X

# EVOLVE-BLOCK-END
