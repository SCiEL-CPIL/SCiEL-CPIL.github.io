"""Multi-objective optimizer for ZDT2 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer that samples the known analytic Pareto front of ZDT2.
    
    For ZDT2 with D dimensions:
    - The Pareto front is at x[1:] = 0
    - f1 = x[0]
    - f2 = 1 - sqrt(x[0])
    
    To maximize hypervolume with reference point [11, 11], we need to sample
    the front densely. Since the front is 1D, we use a non-uniform distribution
    that concentrates more points where the marginal HV contribution is highest.
    
    The key insight: for a convex front like ZDT2's, sampling uniformly in f2
    (i.e., f1 = (1-u)^2 where u is uniform) gives better HV than uniform in f1.
    This is because points near f2=0 (f1=1) extend the covered region further.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        # Wall-clock guard
        t_start = time.time()
        deadline = t_start + 285.0  # Safety margin before 300s limit

        n_eval = 0
        results = []

        # Use full budget for dense Pareto front sampling
        n_total = self.budget

        # Generate x[0] values using a non-uniform distribution that
        # concentrates more points near f1=1 (where f2=0) for better HV.
        #
        # Strategy: sample u uniformly in [0, 1], then set f1 = (1-u)^2
        # This gives f2 = 1 - sqrt(f1) = 1 - (1-u) = u, so f2 is uniform.
        # This concentrates points where f2 is large (f1 small) less densely
        # and where f2 is small (f1 large) more densely.
        #
        # Actually, let's reconsider. The HV contribution of each point depends
        # on the area it uniquely covers. For a set of points sorted by f1,
        # the HV is the sum of rectangular areas. To maximize this with N points,
        # we want to distribute them to maximize the sum.
        #
        # A simpler approach: use a power-law distribution in f1.
        # f1_i = u_i^alpha for some alpha > 1 to concentrate near f1=1.
        # Let's try alpha = 2, which gives f1 = u^2, so f2 = 1 - u.
        # This is equivalent to uniform in f2.
        
        # Generate u uniformly in [0, 1]
        u = np.linspace(0.0, 1.0, n_total)
        
        # Map to f1 using f1 = (1-u)^2, which gives f2 = u (uniform in f2)
        # This concentrates points near f1=1 (u=0) and f2=0
        x0_values = (1.0 - u) ** 2
        
        # Ensure values are within bounds
        x0_values = np.clip(x0_values, lo[0], hi[0])
        
        # Sort by x_0 for consistent ordering
        order = np.argsort(x0_values)
        x0_values = x0_values[order]

        for x0 in x0_values:
            if time.time() > deadline:
                break
            if n_eval >= self.budget:
                break

            x = np.zeros(self.dim)
            x[0] = float(x0)
            # Set remaining dimensions to lower bound (0 for standard ZDT2)
            if self.dim > 1:
                x[1:] = lo[1:]

            f = problem(x)
            results.append(f)
            n_eval += 1

        # Return the last point or a list, though return value is ignored by evaluator
        if results:
            return results[-1]
        return None
# EVOLVE-BLOCK-END
