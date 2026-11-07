"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using an evolutionary strategy with deadline guard
    and tracked archive for hypervolume computation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Initialize population
        pop_size = min(20, self.budget // 4)
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_X = list(X)
        all_F = list(F)

        def is_dominated(F, i):
            """Check if solution i is dominated by any other in F."""
            fi = F[i]
            for j in range(len(F)):
                if j != i:
                    fj = F[j]
                    if np.all(fj <= fi) and np.any(fj < fi):
                        return True
            return False

        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Find non-dominated solutions in current population
            dominated = np.array([is_dominated(F, i) for i in range(len(F))])
            non_dominated = ~dominated
            if not np.any(non_dominated):
                non_dominated = np.ones(len(F), dtype=bool)

            # Select parent from non-dominated set with bias toward better solutions
            nd_indices = np.where(non_dominated)[0]
            if len(nd_indices) > 1:
                # Weight by inverse of objective sum (lower sum = better)
                obj_sums = np.sum(F[nd_indices], axis=1)
                weights = 1.0 / (obj_sums + 1e-10)
                weights /= weights.sum()
                parent_idx = nd_indices[rng.choice(len(nd_indices), p=weights)]
            else:
                parent_idx = nd_indices[0]
            parent = X[parent_idx]

            # Adaptive step size: start larger, shrink over time
            progress = used / self.budget
            base_step = 0.3 * (1.0 - 0.7 * progress)
            step_size = base_step
            child = parent + step_size * range_ * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Record point
            all_X.append(child)
            all_F.append(fc)

            # Replace worst solution: find the solution dominated by fc, or worst by sum
            worst_idx = None
            for i in range(len(F)):
                if np.all(fc <= F[i]) and np.any(fc < F[i]):
                    worst_idx = i
                    break
            if worst_idx is None:
                # Pick the solution with highest objective sum among dominated-by-nothing
                # or just the worst overall
                worst_idx = int(np.argmax(np.sum(F, axis=1)))

            X[worst_idx] = child
            F[worst_idx] = fc

        return all_X
# EVOLVE-BLOCK-END
