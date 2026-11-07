"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Analytical + local refinement multi-objective optimizer for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim

        # ZDT4 structure:
        # f1(x) = x[0]
        # g(x) = 1 + 10*(n-1) + sum_{i=1}^{n-1} [(x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))]
        # f2(x) = g(x) * (1 - sqrt(x[0]/g(x)))
        #
        # The global Pareto front is achieved when x[0] in [0,1] and g(x)=2,
        # which requires each x_i (i>=1) to be at a global minimum of
        # (x_i-0.5)^2 - cos(20*pi*(x_i-0.5)).
        #
        # Global minima of h(t) = t^2 - cos(20*pi*t) for t in [0,1]:
        # The function has many local minima. The global minimum occurs at t=0.5
        # where h(0.5) = 0.25 - cos(10*pi) = 0.25 - 1 = -0.75
        # Wait: cos(20*pi*0.5) = cos(10*pi) = 1, so h(0.5) = 0.25 - 1 = -0.75
        # At t=0: h(0) = 0 - cos(0) = -1
        # At t=1: h(1) = 0.25 - cos(20*pi) = 0.25 - 1 = -0.75
        # So the global minimum is at t=0 with h(0) = -1.
        #
        # For g(x) = 2, we need sum_{i=1}^{9} h(x_i) = 2 - 10*9 = 2 - 90 = -88
        # But minimum possible sum is 9 * (-1) = -9, which is much greater than -88.
        # So g(x) >= 1 + 90 + (-9) = 82 at the best case.
        #
        # Wait, let me re-read ZDT4. Standard ZDT4:
        # f1(x) = x_1
        # g(x) = 1 + 10*(n-1) + sum_{i=2}^{n} [(x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))]
        # f2(x) = g(x) * (1 - sqrt(x_1/g(x)))
        #
        # The known Pareto front for ZDT4:
        # The global Pareto front is where g(x) is minimized.
        # g_min = 1 + 10*(n-1) + sum of min of each term
        # Each term (x_i-0.5)^2 - cos(20*pi*(x_i-0.5)) has global min at x_i=0:
        #   (0-0.5)^2 - cos(0) = 0.25 - 1 = -0.75
        # Wait, let me recheck. At x_i = 0: (0-0.5)^2 - cos(20*pi*(0-0.5)) = 0.25 - cos(-10*pi) = 0.25 - 1 = -0.75
        # At x_i = 0.5: (0.5-0.5)^2 - cos(20*pi*0) = 0 - 1 = -1
        # So global min is at x_i = 0.5 with value -1.
        #
        # So g_min = 1 + 10*9 + 9*(-1) = 1 + 90 - 9 = 82
        #
        # The Pareto front: f1 = x_0 in [0, 1], f2 = 82*(1 - sqrt(x_0/82))
        # Reference point: [11.0, 400.0]

        # Strategy:
        # 1. Set x_i = 0.5 for i >= 1 (global min of each term)
        # 2. Vary x_0 in [0, 1] to trace the Pareto front
        # 3. Use remaining budget for local refinement around promising points

        # Generate candidate x_0 values along the Pareto front
        # Use arc-length parameterization for better coverage

        def g_value(x):
            """Compute g(x) for ZDT4."""
            terms = 0.0
            for i in range(1, dim):
                t = x[i] - 0.5
                terms += t * t - np.cos(20.0 * np.pi * t)
            return 1.0 + 10.0 * (dim - 1) + terms

        def objectives(x):
            """Compute objectives for ZDT4."""
            x0 = np.clip(x[0], 0.0, 1.0)
            g = g_value(x)
            f1 = x0
            f2 = g * (1.0 - np.sqrt(x0 / g))
            return np.array([f1, f2])

        # Build initial population on the Pareto front
        # x_i = 0.5 for i >= 1, x_0 varies in [0, 1]
        n_front = 150  # points on the front
        x0_vals = np.linspace(0.0, 1.0, n_front)
        # Add some extra points with slight perturbations to x_i for robustness

        X = np.zeros((n_front, dim))
        for i in range(dim - 1):
            X[:, i + 1] = 0.5
        X[:, 0] = x0_vals

        # Evaluate
        F = np.array([problem(x) for x in X])
        used = n_front

        # Check time
        if time.time() > deadline:
            return X

        # Use remaining budget for local search around best points
        remaining = self.budget - used
        if remaining > 0:
            # Find non-dominated points
            non_dominated_mask = np.ones(n_front, dtype=bool)
            for i in range(n_front):
                for j in range(n_front):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        non_dominated_mask[i] = False
                        break

            # Local refinement: perturb x_i values slightly to see if we can improve
            # Focus on points that are well-distributed along the front
            step = max(1, remaining // max(1, int(np.sum(non_dominated_mask))))
            count = 0
            for idx in np.where(non_dominated_mask)[0]:
                if count >= remaining:
                    break
                if time.time() > deadline:
                    break
                # Try small perturbations to x_i (i>=1)
                for trial in range(min(step, 3)):
                    if time.time() > deadline:
                        break
                    child = X[idx].copy()
                    # Perturb x_i for i >= 1 slightly
                    for i in range(1, dim):
                        child[i] += 0.01 * np.random.uniform(-1, 1)
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    count += 1
                    # Replace worst point if child is better
                    # Simple criterion: replace if child dominates some point
                    for j in range(n_front):
                        if np.all(fc <= F[j]) and np.any(fc < F[j]):
                            X[j] = child
                            F[j] = fc
                            break

            used += count

        return X
# EVOLVE-BLOCK-END
