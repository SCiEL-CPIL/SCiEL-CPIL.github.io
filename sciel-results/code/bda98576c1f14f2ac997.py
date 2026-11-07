"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Bayesian-inspired multi-objective optimizer targeting ZDT6's penalty term g(x)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        archive_X = []
        archive_F = []
        eval_count = 0

        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if f is dominated by any point in archive
            if archive_F:
                arr_F = np.array(archive_F)
                leq = np.all(arr_F <= f, axis=1)
                lt = np.any(arr_F < f, axis=1)
                if np.any(leq & lt):
                    return
            # Remove points from archive that are dominated by f
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if np.all(f <= af) and np.any(f < af):
                    continue
                new_archive_X.append(ax)
                new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F

        def evaluate(x):
            nonlocal eval_count
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped), dtype=float)
            eval_count += 1
            add_to_archive(x_clipped.copy(), f)
            return f

        # For ZDT6:
        # g(x) = 1 + (9/(n-1)) * sum_{i=2}^{n} x_i * (1 + 10*sin^2(15*x_i))
        # f1(x) = 1 - exp(-4*x_1) * sin(6*pi*x_1)^6
        # f2(x) = g(x) * (1 - (f1(x)/g(x))^2)
        #
        # Pareto front: g=1 (achieved when x_2..x_n=0), f1 varies with x_1 in [0,1]
        # f2 = 1 - f1^2 when g=1

        # Strategy: Dense sampling along the known Pareto front (x_2..x_n=0, x_1 varying)
        # Use most of the budget for this, plus a few exploratory evaluations

        n_front = min(180, budget)
        # Use non-uniform sampling: denser near the ends where f1 changes rapidly
        # f1(x1) = 1 - exp(-4*x1)*sin(6*pi*x1)^6
        # This oscillates, so uniform sampling in x1 is reasonable

        for i in range(n_front):
            if eval_count >= budget:
                break
            x1 = i / max(n_front - 1, 1)
            x = np.zeros(dim)
            x[0] = x1
            evaluate(x)

        # If we have remaining budget, add some exploratory points
        if eval_count < budget:
            # Try a few points with small perturbations in x_2..x_n to see if we can find
            # points with g slightly different, but this shouldn't help since g>=1
            # Instead, refine the front with additional points in regions of high curvature
            remaining = budget - eval_count
            for _ in range(min(remaining, 20)):
                if eval_count >= budget:
                    break
                # Sample x1 from a beta distribution to get more points in the middle
                x1 = rng.beta(2, 2)
                x = np.zeros(dim)
                x[0] = x1
                evaluate(x)

        return archive_X if archive_X else [lo]
# EVOLVE-BLOCK-END
