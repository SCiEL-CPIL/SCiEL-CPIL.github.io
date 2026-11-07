"""Multi-objective optimizer for ZDT1 using a structured evolutionary approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective optimizer with ZDT1-specific structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # ZDT1 structure: x1 is the "shape" variable, x2..x30 are "aggregation" variables
        # g(x) = 1 + 10/n * sum(x_i for i=2..n)
        # f1 = x1, f2 = g(x) * (1 - sqrt(f1/g(x)))
        # Pareto front: g=1 (i.e., sum(x2..x30)=0), f2 = 1 - sqrt(f1)
        # So optimal x2..x30 are all 0, and x1 should sweep [0,1]

        # Strategy:
        # 1. Initialize a diverse population
        # 2. Use a mutation strategy that exploits ZDT1 structure
        # 3. Maintain diversity in x1 while pushing x2..x30 toward 0

        pop_size = 50
        # Initial population: diverse in x1, random in x2..x30
        X = np.zeros((pop_size, self.dim))
        X[:, 0] = np.linspace(0, 1, pop_size)  # diverse x1
        X[:, 1:] = rng.random((pop_size, self.dim - 1))

        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Return True if a dominates b (both objectives minimized)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Simple non-dominated sort, returns list of fronts."""
            fronts = []
            remaining = list(range(len(F)))
            while remaining:
                front = []
                for i in remaining:
                    is_dominated = False
                    for j in remaining:
                        if i != j and dominates(F[j], F[i]):
                            is_dominated = True
                            break
                    if not is_dominated:
                        front.append(i)
                fronts.append(front)
                remaining = [i for i in remaining if i not in front]
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return np.ones(n) * np.inf
            cd = np.zeros(n)
            for m in range(self.n_obj):
                sorted_idx = np.argsort(F[indices, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F[indices[sorted_idx[-1]], m] - F[indices[sorted_idx[0]], m]
                if f_range > 0:
                    for k in range(1, n - 1):
                        cd[sorted_idx[k]] += (F[indices[sorted_idx[k + 1]], m] - F[indices[sorted_idx[k - 1]], m]) / f_range
            return cd

        # Main loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Get non-dominated fronts
            fronts = non_dominated_sort(F)

            # Select parents from first front (or first two if needed)
            parent_pool = fronts[0]
            if len(parent_pool) < 2:
                parent_pool = fronts[0] + fronts[1] if len(fronts) > 1 else fronts[0]

            # Generate children with ZDT1-aware mutation
            n_children = min(10, self.budget - used)
            for _ in range(n_children):
                # Pick two parents
                p1_idx = parent_pool[rng.integers(len(parent_pool))]
                p2_idx = parent_pool[rng.integers(len(parent_pool))]

                # Crossover: blend
                alpha = rng.random()
                child_x1 = alpha * X[p1_idx, 0] + (1 - alpha) * X[p2_idx, 0]
                child_x1 = np.clip(child_x1, lo[0], hi[0])

                # For x2..x30: bias toward 0 (since g is minimized when these are 0)
                # Use a mutation that pushes toward 0 with some randomness
                child_rest = np.minimum(X[p1_idx, 1:], X[p2_idx, 1:])  # take the smaller (better for g)
                # Add small random perturbation, biased toward 0
                noise = rng.random(self.dim - 1) * 0.05
                child_rest = np.clip(child_rest - noise, lo[1:], hi[1:])

                child = np.concatenate([[child_x1], child_rest])
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Replace worst individual
                # Compute a simple fitness: sum of objectives
                F_sum = np.sum(F, axis=1)
                worst = int(np.argmax(F_sum))
                X[worst] = child
                F[worst] = fc

            # Occasionally do a targeted search: set x2..x30 to near 0 and vary x1
            if used % 20 < n_children:
                for _ in range(n_children):
                    if time.time() > deadline:
                        break
                    x1_val = rng.random()
                    x_rest = rng.random(self.dim - 1) * 0.01  # very small values
                    child = np.concatenate([[x1_val], x_rest])
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    used += 1
                    F_sum = np.sum(F, axis=1)
                    worst = int(np.argmax(F_sum))
                    X[worst] = child
                    F[worst] = fc

        return X
# EVOLVE-BLOCK-END
