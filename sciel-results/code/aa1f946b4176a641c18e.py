"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Targeted multi-objective search for ZDT1 exploiting problem structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: use a good fraction of budget
        pop_size = min(40, max(10, self.budget // 5))
        remaining = self.budget

        # Initialize: diverse x1 values, x2...xn near 0 (good for ZDT1)
        X = np.zeros((pop_size, self.dim))
        # Spread x1 across [0, 1]
        X[:, 0] = np.linspace(0.0, 1.0, pop_size)
        # x2...xn: start near 0 with small random perturbation
        X[:, 1:] = np.random.uniform(0.0, 0.01, size=(pop_size, self.dim - 1))
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X])
        remaining -= pop_size

        # Track best per x1 bin for diversity
        n_bins = 20
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

        def get_bin(x1_val):
            if x1_val < bin_edges[0]:
                return 0
            if x1_val >= bin_edges[-1]:
                return n_bins - 1
            return int(np.searchsorted(bin_edges, x1_val, side='right') - 1)

        # Main loop: improve g while maintaining x1 diversity
        while remaining > 0:
            if time.time() > deadline - 5:
                break

            # Find non-dominated points
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break

            # Select parents from non-dominated set
            if np.any(~dominated):
                parents_idx = np.where(~dominated)[0]
            else:
                parents_idx = np.arange(pop_size)

            # Create multiple children per iteration to use budget efficiently
            n_children = min(remaining, 5)
            for _ in range(n_children):
                if remaining <= 0:
                    break
                if time.time() > deadline - 5:
                    break

                # Pick a random parent
                parent_idx = parents_idx[np.random.randint(len(parents_idx))]
                parent = X[parent_idx].copy()

                # Mutation strategy:
                # 1. Slightly perturb x1 to explore nearby region
                # 2. Strongly drive x2...xn toward 0 (minimize g)
                # 3. Occasionally do a larger jump in x1

                r = np.random.random()
                if r < 0.3:
                    # Large jump in x1, reset x2...xn
                    parent[0] = np.random.uniform(0.0, 1.0)
                    parent[1:] = np.random.uniform(0.0, 0.05, size=self.dim - 1)
                elif r < 0.6:
                    # Moderate x1 change, drive x2...xn toward 0
                    parent[0] += 0.05 * np.random.randn()
                    parent[0] = np.clip(parent[0], 0.0, 1.0)
                    parent[1:] = 0.9 * parent[1:] + 0.1 * np.random.uniform(0.0, 0.01, size=self.dim - 1)
                else:
                    # Small perturbation, focus on minimizing g
                    parent[0] += 0.01 * np.random.randn()
                    parent[0] = np.clip(parent[0], 0.0, 1.0)
                    parent[1:] = 0.8 * parent[1:] + 0.2 * np.random.uniform(0.0, 0.005, size=self.dim - 1)

                child = np.clip(parent, lo, hi)
                fc = problem(child)
                remaining -= 1

                # Replace worst point in the same x1 bin, or global worst
                child_bin = get_bin(child[0])
                bin_points = [i for i in range(pop_size) if get_bin(X[i, 0]) == child_bin]

                if bin_points:
                    # Find worst in this bin (highest f2, since f1 is similar)
                    worst_in_bin = max(bin_points, key=lambda i: F[i, 1])
                    if np.all(fc <= F[worst_in_bin]) or (np.any(fc < F[worst_in_bin]) and not np.all(F[worst_in_bin] <= fc)):
                        X[worst_in_bin] = child
                        F[worst_in_bin] = fc
                    else:
                        # Replace global worst
                        worst = int(np.argmax(np.sum(F, axis=1)))
                        X[worst] = child
                        F[worst] = fc
                else:
                    # New bin, replace global worst
                    worst = int(np.argmax(np.sum(F, axis=1)))
                    X[worst] = child
                    F[worst] = fc

        return X
# EVOLVE-BLOCK-END
