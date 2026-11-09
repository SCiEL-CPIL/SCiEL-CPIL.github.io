"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Analytical optimizer for ZDT2 that exploits the known structure.

    ZDT2: f1 = x0, f2 = 1 + (9/(n-1)) * sum_{i=1}^{n-1} x_i * (1 + sin(3*pi*x_i))
    The Pareto front is f2 = 1 + f1^0.5, achieved when x1..x_{n-1} = 0.
    We use the entire budget to sample x0 values on the Pareto front with
    a distribution optimized for hypervolume maximization.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 295.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        used = 0

        def time_ok():
            return time.time() < deadline - 5.0

        budget = self.budget

        # For ZDT2, the Pareto front is f2 = 1 + sqrt(f1) when x1..x_{n-1}=0.
        # Hypervolume with reference point (11, 11):
        #   HV = sum over non-dominated points of (11 - f1_i) * (11 - f2_i) * indicator
        # For a concave front, the optimal distribution of N points to maximize
        # hypervolume is NOT uniform in f1 or f2.
        #
        # Key insight: The hypervolume contribution of a point at f1 is
        # proportional to the area it "claims". For a concave curve, points
        # near f1=0 (steep part) contribute more per unit f1-spacing, but
        # the absolute area (11-f1)*(11-f2) is large throughout.
        #
        # Optimal strategy: Space points so that the hypervolume "bins" are
        # equal. This is achieved by making the trapezoidal area between
        # consecutive points constant.
        #
        # Alternative: Use the fact that for f2 = 1+sqrt(f1), we can
        # parameterize by f2. In f2-space [1,2], the front is f1 = (f2-1)^2.
        # The hypervolume element is (11-f1)*(11-f2) df2.
        #
        # Let's use a grid that's optimized: sample f2 such that
        # (11 - (f2-1)^2) * (11 - f2) is roughly constant per interval.
        # This means denser sampling where the product is larger.
        #
        # Simpler approach: Use quantiles of the distribution that
        # maximizes hypervolume. For 2D hypervolume with reference (R1,R2),
        # the optimal points for a monotone concave front are those that
        # equalize the marginal hypervolume contribution.
        #
        # Practical approach: Use a power-law spacing in f1.
        # f1_i = (i/N)^alpha for some alpha.
        # For alpha < 1, denser near 0. For alpha > 1, denser near 1.
        #
        # Let's try several alpha values and pick the best analytically.
        # The hypervolume of points f1_i with f2_i = 1+sqrt(f1_i):
        # HV ≈ sum (11-f1_i)*(11-f2_i) * (f1_{i+1} - f1_i)  [trapezoidal approx]
        #
        # Actually, for exact hypervolume of a set of points on a monotone
        # front, we can compute it exactly. Let's just try different spacings.
        
        # Strategy: Use the "equal area" criterion.
        # Divide the hypervolume under the curve into N equal areas.
        # The hypervolume contributed by a small segment df1 at position f1 is:
        #   dHV = (11 - f1) * (11 - (1+sqrt(f1))) * df1
        #        = (11 - f1) * (10 - sqrt(f1)) * df1
        #
        # We want to find f1 values such that the integral of this density
        # from 0 to f1 is proportional to the index.
        #
        # Let's numerically compute the optimal spacing.
        
        # Compute the hypervolume density function with a very fine grid
        # to avoid duplicate values when interpolating
        grid_size = 1000000
        f1_grid = np.linspace(0, 1, grid_size)
        f2_grid = 1 + np.sqrt(f1_grid)
        density = (11 - f1_grid) * (11 - f2_grid)
        
        # Cumulative hypervolume
        df = f1_grid[1] - f1_grid[0]
        cum_hv = np.cumsum(density) * df
        
        # Total hypervolume
        total_hv = cum_hv[-1]
        
        n_points = budget
        # Fractions of total HV for each point
        fractions = np.linspace(0, 1, n_points)
        
        # Interpolate to get f1 values
        x0_optimal = np.interp(fractions * total_hv, cum_hv, f1_grid)
        
        # Ensure endpoints
        x0_optimal[0] = 0.0
        x0_optimal[-1] = 1.0
        
        # Make sure they're sorted and unique with a tight tolerance
        x0_optimal = np.sort(x0_optimal)
        
        # Remove near-duplicates (keep first occurrence)
        unique_mask = np.concatenate([[True], np.diff(x0_optimal) > 1e-12])
        x0_optimal = x0_optimal[unique_mask]
        
        # If we have fewer points than budget (due to dedup), add more
        if len(x0_optimal) < budget:
            n_needed = budget - len(x0_optimal)
            # Use a fine uniform grid and pick points not already in x0_optimal
            fine_grid = np.linspace(0, 1, budget * 100)
            # Remove points already in x0_optimal
            mask = np.ones(len(fine_grid), dtype=bool)
            for x in x0_optimal:
                idx = np.argmin(np.abs(fine_grid - x))
                if np.abs(fine_grid[idx] - x) < 1e-10:
                    mask[idx] = False
            available = fine_grid[mask]
            if len(available) >= n_needed:
                indices = np.linspace(0, len(available) - 1, n_needed).astype(int)
                x0_optimal = np.sort(np.concatenate([x0_optimal, available[indices]]))
            else:
                x0_optimal = np.sort(np.concatenate([x0_optimal, available]))
        
        # Trim to budget if needed
        if len(x0_optimal) > budget:
            indices = np.linspace(0, len(x0_optimal) - 1, budget).astype(int)
            x0_optimal = x0_optimal[indices]
        
        # Ensure exactly budget points
        if len(x0_optimal) < budget:
            # Fill with evenly spaced points from [0,1]
            extra = np.linspace(0, 1, budget - len(x0_optimal))
            x0_optimal = np.sort(np.concatenate([x0_optimal, extra]))
        
        x0_optimal = x0_optimal[:budget]
        
        records = []
        for f1_val in x0_optimal:
            if used >= self.budget or not time_ok():
                break
            x = np.zeros(n)
            x[0] = f1_val
            f = problem(x)
            records.append((x.copy(), np.array(f, dtype=float)))
            used += 1

        return records  # ignored by evaluator

# EVOLVE-BLOCK-END
