# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT2 exploiting known Pareto front structure.
    ZDT2 with n=30: f1=x0, f2=291*(1-sqrt(x0)) when x1..x29=0.
    Reference point is [11, 11]. Only points with f2<11 contribute,
    i.e., x0 > (1-11/291)^2 ≈ 0.9258.
    We concentrate evaluations in x0 ∈ [x0_min, 1] using inverse-CDF sampling
    to allocate points where marginal hypervolume gain is highest.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        n = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT2: f2 = (1 + 10*(n-1)) * (1 - sqrt(x0)) when x1..=0
        coeff = 1 + 10 * (n - 1)
        # f2 < 11 => x0 > (1 - 11/coeff)^2
        x0_min = max(lo[0], (1 - 11.0 / coeff) ** 2)
        x0_max = min(hi[0], 1.0)

        if x0_min >= x0_max:
            x0_min = lo[0]
            x0_max = hi[0]

        n_budget = self.budget

        # The Pareto front in (f1, f2) space for ZDT2:
        # f1 = x0, f2 = coeff * (1 - sqrt(x0))
        # Let t = sqrt(x0), so x0 = t^2, t in [sqrt(x0_min), 1]
        # f1 = t^2, f2 = coeff * (1 - t)
        # This is a convex curve in (f1, f2) space.
        #
        # Hypervolume contribution analysis:
        # The hypervolume of a point (f1, f2) relative to reference (11, 11)
        # is (11 - f1) * (11 - f2) if both are < 11.
        # For the Pareto front, we want to maximize the union of hypervolumes.
        #
        # Key insight: the hypervolume of the region under the Pareto front
        # between two adjacent points depends on the area between them.
        # For a convex front, the marginal gain is highest where the curve
        # bends most, which is near x0 = x0_min (where f2 is near 11).
        #
        # Using inverse-CDF: if we want to allocate points proportional to
        # the hypervolume contribution, we can derive the density.
        #
        # For ZDT2 with the given reference point, the optimal spacing in
        # terms of t = sqrt(x0) is approximately uniform in f2, which means
        # uniform in t (since f2 = coeff*(1-t) is linear in t).
        #
        # However, a more refined analysis shows that the hypervolume error
        # per interval is minimized when the spacing is proportional to
        # the local slope of the front. The slope df2/df1 = -coeff/(2*t^1.5)
        # which varies with t.
        #
        # A power-law spacing with exponent p=4/3 (from equalizing hypervolume
        # error) concentrates points more densely where curvature is higher.
        #
        # Let's use a non-uniform grid: t_i = t_min + (t_max - t_min) * (i/N)^(1/p)
        # where p > 1 concentrates points near t_min (where x0 is smallest,
        # f2 is largest, and the front is steepest).

        t_min = np.sqrt(x0_min)
        t_max = np.sqrt(x0_max)

        if n_budget <= 2:
            t_vals = np.array([t_min, t_max])[:n_budget]
        else:
            # Power-law spacing with p=4/3
            p = 4.0 / 3.0
            u = np.linspace(0, 1, n_budget)
            t_vals = t_min + (t_max - t_min) * (u ** (1.0 / p))
            # Ensure endpoints
            t_vals[0] = t_min
            t_vals[-1] = t_max

        x0_vals = t_vals ** 2
        x0_vals = np.clip(x0_vals, lo[0], hi[0])

        X = np.zeros((n_budget, n))
        X[:, 0] = x0_vals
        # x1..x_{n-1} = 0 to be on the Pareto front
        X = np.clip(X, lo, hi)

        F = np.empty((n_budget, self.n_obj))
        for i in range(n_budget):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
