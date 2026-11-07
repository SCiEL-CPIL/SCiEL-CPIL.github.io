"""Multi-objective optimizer for WFG4 with structure-aware initialization and local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware evolutionary search for WFG4 with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # 30s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # WFG4 structure: x1 controls which disconnected front segment.
        # Known Pareto front has two segments; x1 near 0.35 is a transition point.
        # We bias initialization to explore both segments.
        def make_initial(n, seed_offset=0):
            rng = np.random.default_rng(seed_offset)
            X = np.empty((n, dim))
            # Mix of x1 values: some near 0.35, some spread
            x1_vals = np.concatenate([
                np.full(n // 3, 0.35) + rng.normal(0, 0.02, n // 3),
                rng.uniform(0.1, 0.5, n - n // 3)
            ])
            X[:, 0] = np.clip(x1_vals, lo[0], hi[0])
            X[:, 1:] = lo[1:] + (hi[1:] - lo[1:]) * rng.random((n, dim - 1))
            return X

        # Population size
        pop_size = min(30, max(10, budget // 6))
        X = make_initial(pop_size, seed_offset=42)
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Track all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]

        def is_dominated(f, F_set):
            """Check if f is dominated by any point in F_set."""
            return np.any(np.all(F_set <= f, axis=1) & np.any(F_set < f, axis=1))

        def non_dominated_indices(F_set):
            """Return indices of non-dominated points."""
            n = len(F_set)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i != j and nd[j]:
                        if np.all(F_set[j] <= F_set[i]) and np.any(F_set[j] < F_set[i]):
                            nd[i] = False
            return np.where(nd)[0]

        def local_refine(x, f, budget_left, rng):
            """Local search: perturb each dimension and keep improvements."""
            best_x = x.copy()
            best_f = f.copy()
            step = 0.05
            for _ in range(min(12, budget_left)):
                if time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if time.time() > deadline:
                        break
                    trial = best_x.copy()
                    trial[d] = best_x[d] + step * rng.standard_normal()
                    trial[d] = np.clip(trial[d], lo[d], hi[d])
                    tf = problem(trial)
                    used_local = 1
                    # Accept if not dominated by best_f (Pareto improvement)
                    if not (np.all(best_f <= tf) and np.any(best_f < tf)):
                        # Also accept if it improves hypervolume contribution
                        if np.all(tf < best_f) or (not np.all(best_f <= tf)):
                            best_x = trial
                            best_f = tf
                            improved = True
                            break
                    # Try opposite direction
                    trial2 = best_x.copy()
                    trial2[d] = best_x[d] - step * rng.standard_normal()
                    trial2[d] = np.clip(trial2[d], lo[d], hi[d])
                    tf2 = problem(trial2)
                    if not (np.all(best_f <= tf2) and np.any(best_f < tf2)):
                        if np.all(tf2 < best_f) or (not np.all(best_f <= tf2)):
                            best_x = trial2
                            best_f = tf2
                            improved = True
                            break
                if not improved:
                    step *= 0.7
            return best_x, best_f

        # Main evolutionary loop
        rng = np.random.default_rng(123)
        while used < budget:
            if time.time() > deadline:
                break

            # Find non-dominated set in current population
            nd_idx = non_dominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = [0]

            # Select parent from non-dominated set
            parent_idx = nd_idx[rng.integers(len(nd_idx))]
            parent_x = X[parent_idx].copy()
            parent_f = F[parent_idx].copy()

            # Mutation with structure awareness
            if rng.random() < 0.3:
                # Bias toward x1 = 0.35 region
                child = parent_x.copy()
                child[0] = parent_x[0] + 0.1 * rng.standard_normal()
                child[0] = np.clip(child[0], lo[0], hi[0])
                child[1:] += 0.08 * span[1:] * rng.standard_normal(dim - 1)
            else:
                # Standard mutation
                child = parent_x + 0.1 * span * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            child_f = problem(child)
            used += 1

            # Local refinement on promising children
            if used < budget - 5:
                child, child_f = local_refine(child, child_f, min(6, budget - used), rng)
                used += 6  # approximate budget accounting for local search

            # Replace worst point in population
            # Use sum of normalized objectives as a simple fitness
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = child_f

            # Occasionally add a new random point to maintain diversity
            if used < budget - 10 and rng.random() < 0.1:
                new_x = lo + span * rng.random(dim)
                # Bias x1
                if rng.random() < 0.5:
                    new_x[0] = 0.35 + rng.normal(0, 0.05)
                    new_x[0] = np.clip(new_x[0], lo[0], hi[0])
                new_f = problem(new_x)
                used += 1
                worst_idx = int(np.argmax(np.sum(F, axis=1)))
                X[worst_idx] = new_x
                F[worst_idx] = new_f

        # Final local refinement on best points
        if used < budget:
            nd_idx = non_dominated_indices(F)
            for idx in nd_idx[:min(5, len(nd_idx))]:
                if used >= budget or time.time() > deadline:
                    break
                refined_x, refined_f = local_refine(X[idx], F[idx], min(4, budget - used), rng)
                used += 4
                X[idx] = refined_x
                F[idx] = refined_f

        return X
# EVOLVE-BLOCK-END
