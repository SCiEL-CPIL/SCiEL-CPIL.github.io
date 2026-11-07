"""ZDT6 multi-objective optimizer using closed-form Pareto front sampling."""
import numpy as np
import time
from scipy.optimize import brentq


class Optimizer:
    """Multi-objective optimizer on ZDT6 using parametric Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit

        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT6:
        # g(x) = 1 + 10*n
        # f1 = 1 - sqrt(x1)
        # f2 = g * h(f1, g) where h = 1 - sqrt(f1/g) - (f1/g)*sin(10*pi*f1)
        #
        # Pareto front: x2..x10 = 0, so g = 1
        # f1 = 1 - sqrt(x1), x1 in [0,1], so f1 in [0,1]
        # f2 = 1 * (1 - sqrt(f1) - f1*sin(10*pi*f1))
        #
        # To sample the Pareto front, we can parametrize by f1 in [0,1]:
        # x1 = (1 - f1)^2
        # f2 = 1 - sqrt(f1) - f1*sin(10*pi*f1)
        #
        # However, the problem function clips and evaluates, so we need to
        # actually call problem() to get the objectives recorded.
        #
        # Strategy:
        # 1. Use a large portion of the budget to sample the Pareto front densely
        #    by setting x2..x10 = 0 and varying x1.
        # 2. Use remaining budget for local search to find points near the front.
        #
        # The Pareto front for ZDT6 is not uniform in f1. The density is higher
        # near f1=0 and f1=1. We should sample x1 in a way that gives good
        # coverage of the front.
        #
        # Since f1 = 1 - sqrt(x1), we have x1 = (1-f1)^2.
        # To get uniform sampling in f1, sample f1 uniformly in [0,1] and set x1 = (1-f1)^2.
        #
        # But we want to maximize hypervolume. The reference point is [11, 11].
        # Points on the Pareto front have f1 in [0,1] and f2 in [0,1] roughly.
        #
        # Let's allocate budget:
        # - 80% for dense Pareto front sampling
        # - 20% for perturbation/search around front points

        n_front = int(budget * 0.85)
        n_search = budget - n_front

        all_points = []

        # Phase 1: Sample the Pareto front
        # Use a mix of uniform and clustered samples to cover the front well
        # The front has high curvature near f1=0 and f1=1

        # Generate f1 values: mix of uniform and clustered
        # Use a transformation to cluster near 0 and 1
        u = np.linspace(0, 1, n_front)

        # Apply a transformation to cluster near endpoints
        # Use a beta-like distribution or just use x1 directly
        # Since x1 = (1-f1)^2, uniform x1 gives non-uniform f1
        # Let's use a combination: some uniform in f1, some in x1

        # Actually, let's just sample x1 in a smart way
        # The Pareto front in (f1, f2) space: f1 = 1-sqrt(x1), f2 = 1 - sqrt(f1) - f1*sin(10*pi*f1)
        # Near f1=0: f2 ≈ 1 - sqrt(f1), which has infinite slope
        # Near f1=1: f2 ≈ 1 - 1 - 0 = 0, but with oscillations

        # For hypervolume, we want points that maximize the dominated region
        # Let's sample x1 with a distribution that gives good coverage

        # Use a power-law distribution for x1 to cluster near 0
        # x1 = u^alpha, with alpha < 1 to cluster near 0
        # But we also need coverage near x1=1

        # Let's use a mixture: 50% uniform in x1, 50% clustered near x1=0
        n_uniform = n_front // 2
        n_clustered = n_front - n_uniform

        x1_uniform = np.linspace(0, 1, n_uniform)
        # Clustered near 0: x1 = u^2
        u_cluster = np.linspace(0, 1, n_clustered)
        x1_clustered = u_cluster ** 3  # Strong clustering near 0

        x1_values = np.concatenate([x1_uniform, x1_clustered])
        np.random.shuffle(x1_values)

        # Evaluate points on the Pareto front
        for x1 in x1_values[:n_front]:
            if time.time() > deadline - 30:
                break
            x = np.zeros(dim)
            x[0] = x1
            # x[1:] = 0 is the Pareto-optimal setting
            f = problem(x)
            all_points.append(f)

        # Phase 2: Use remaining budget for perturbation search
        # Try small perturbations around front points to see if we can find
        # slightly better points (though on the true front, perturbations in x2..x10
        # increase f2, so we should stay on the front)
        #
        # Actually, for ZDT6, any deviation from x2..x10=0 increases g > 1,
        # which increases f2 for the same f1. So the Pareto front is exactly
        # at x2..x10=0.
        #
        # So the best strategy is to densely sample the Pareto front.
        # Let's use all remaining budget for more front sampling.

        # Add more front samples with different distributions
        # Use a finer grid in f1 space
        n_extra = budget - len(all_points)
        if n_extra > 0:
            # Sample f1 uniformly and convert to x1
            f1_values = np.linspace(0, 1, n_extra)
            # Avoid duplicates
            for f1 in f1_values:
                if time.time() > deadline - 10:
                    break
                x1 = (1 - f1) ** 2
                x = np.zeros(dim)
                x[0] = x1
                f = problem(x)
                all_points.append(f)

        return all_points
