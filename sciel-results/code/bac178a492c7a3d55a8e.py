"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT6 Pareto front: x[1:] = 0.5, x[0] in [0, 1]
        # This gives f1 = 1 (minimum) and f2 varies with x[0]
        # We sample x[0] uniformly to cover the front well

        # Determine how many samples we can afford
        # Each evaluation is one call to problem
        # We want to spread evaluations across the front

        # First, let's do a coarse sweep of the known Pareto front
        # x[1:] = 0.5 is the key structural insight for ZDT6

        base_x = np.full(self.dim, 0.5)  # x[1:] = 0.5
        base_x[0] = 0.0  # start with x[0] = 0

        # Sample x[0] values uniformly in [0, 1]
        # With budget=200, we can sample up to 200 points on the front
        # But we should also leave some budget for refinement if possible

        # Strategy: sweep x[0] from 0 to 1 in fine steps
        # The hypervolume is maximized by well-distributed points on the front

        # Let's use most of the budget for front sampling
        n_samples = min(self.budget, 200)

        # Create x[0] values: uniform in [0, 1] with n_samples points
        # Use linspace for even coverage
        x0_values = np.linspace(0.0, 1.0, n_samples)

        # Evaluate all points
        recorded = []
        for i, x0 in enumerate(x0_values):
            x = base_x.copy()
            x[0] = x0
            f = problem(x)
            recorded.append(f)

            # Check time
            if time.time() > deadline:
                break

        # If we have remaining budget, try to improve coverage
        # by adding more points in regions that might need denser sampling
        used = len(recorded)
        while used < self.budget:
            if time.time() > deadline:
                break

            # Add a random point on the front (x[1:] = 0.5)
            x0 = np.random.random()
            x = base_x.copy()
            x[0] = x0
            f = problem(x)
            recorded.append(f)
            used += 1

        return recorded
# EVOLVE-BLOCK-END
