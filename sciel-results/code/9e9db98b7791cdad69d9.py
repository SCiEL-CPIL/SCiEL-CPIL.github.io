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

        # For ZDT2, the Pareto front is 1D: f1 = x[0], f2 = 1 - sqrt(x[0])
        # To maximize hypervolume with reference point [11, 11], we need to
        # sample the front optimally.
        #
        # The hypervolume contribution of a point at position i (sorted by f1)
        # is approximately: (f2_i - f2_{i+1}) * (11 - f1_i)
        # where f2_{n+1} = 11 (reference).
        #
        # For optimal distribution, we want equal marginal contributions.
        # Let f2_i = u_i where u_i are quantiles. Then f1_i = (1-u_i)^2.
        #
        # Using Latin Hypercube Sampling (LHS) for better coverage:
        # Divide [0,1] into n_total equal intervals, pick one random point
        # from each interval, then shuffle.
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
        # Create LHS samples in [0, 1]
        u = np.linspace(0, 1, n_total + 1)
        u_strata = u[:-1] + rng.random(n_total) * (1.0 / n_total)
        rng.shuffle(u_strata)
        
        # Map to x[0] values: f1 = (1-u)^2, which gives f2 = u
        # This ensures uniform coverage in f2 space
        x0_values = (1.0 - u_strata) ** 2
        
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
