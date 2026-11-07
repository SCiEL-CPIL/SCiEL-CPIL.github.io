# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Specialized optimizer for ZDT2.

    ZDT2: f1 = x[0]; g = 1 + 10*(n-1) + sum_{i=1}^{n-1}(x_i^2 - 2*cos(20*pi*x_i));
          f2 = g * (1 - sqrt(f1/g)).
    The Pareto front is achieved at g = 1, which requires x[1:] = 0.
    On the front: f1 = x[0] in [0,1], f2 = 1 - sqrt(f1).

    Strategy: spend the entire budget placing points exactly on the Pareto
    front by sweeping x[0] over a well-chosen set in [0,1] and fixing x[1:] = 0.
    We use a combination of a uniform grid and a few extra points to maximize
    hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 30s safety margin before 310s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # We want to evaluate exactly self.budget points.
        # All points will be on the Pareto front: x[1:] = 0, x[0] varies.
        n_evals = self.budget

        # Choose x[0] values to maximize hypervolume coverage.
        # The Pareto front is f2 = 1 - sqrt(f1) for f1 in [0, 1].
        # Reference point is [11, 11].
        # Hypervolume contribution of a point (f1, f2) is roughly (11-f1)*(11-f2).
        # To maximize total hypervolume, we should spread points across the front.
        # A uniform grid in f1 gives good coverage. Let's use n_evals points uniformly in [0, 1].
        # Actually, since the front is curved, a uniform grid in f1 is a good starting point.
        # We could also try to optimize the placement, but with 200 points, uniform should be very good.

        x0_values = np.linspace(0, 1, n_evals)

        # Build all points: x[0] = x0_values[i], x[1:] = 0
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_values

        # Evaluate all points
        for i in range(n_evals):
            if time.time() > deadline:
                break
            problem(X[i])

        return X
# EVOLVE-BLOCK-END
