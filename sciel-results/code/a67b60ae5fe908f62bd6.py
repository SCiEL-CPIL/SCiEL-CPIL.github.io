"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using an evolutionary strategy with vectorized operations
    and WFG4-aware biased sampling."""

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

        def vectorized_non_dominated(F):
            """Return boolean mask of non-dominated solutions using vectorized ops."""
            n = len(F)
            if n <= 1:
                return np.ones(n, dtype=bool)
            # F[i] is dominated if exists j: F[j] <= F[i] and F[j] != F[i]
            leq = (F[:, np.newaxis, :] <= F[np.newaxis, :, :])  # (n, n, m)
            strict = (F[:, np.newaxis, :] < F[np.newaxis, :, :])  # (n, n, m)
            dom_mask = np.all(leq, axis=2) & np.any(strict, axis=2)  # dom_mask[j,i] = j dominates i
            dominated = np.any(dom_mask, axis=0)
            return ~dominated

        def crowding_distance(F, nd_indices):
            """Compute crowding distance for non-dominated solutions."""
            n_nd = len(nd_indices)
            if n_nd <= 2:
                return np.full(n_nd, np.inf)

            cd = np.zeros(n_nd)
            F_nd = F[nd_indices]

            for m in range(self.n_obj):
                order = np.argsort(F_nd[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                denom = F_nd[order[-1], m] - F_nd[order[0], m]
                if denom == 0:
                    continue
                for k in range(1, n_nd - 1):
                    cd[order[k]] += (F_nd[order[k + 1], m] - F_nd[order[k - 1], m]) / denom

            return cd

        def select_parent(F, X, rng):
            """Tournament selection with non-domination and crowding distance."""
            nd_mask = vectorized_non_dominated(F)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) == 0:
                return X[rng.integers(len(F))]
            if len(nd_indices) == 1:
                return X[nd_indices[0]]

            # Compute crowding distance for non-dominated set
            cd = crowding_distance(F, nd_indices)

            # Tournament: pick 2 random, select one with better crowding distance
            i1, i2 = rng.integers(len(nd_indices), size=2)
            if cd[i1] >= cd[i2]:
                return X[nd_indices[i1]]
            else:
                return X[nd_indices[i2]]

        def wfg4_biased_sample(rng):
            """Generate a sample biased toward the WFG4 Pareto front.
            
            For WFG4, the Pareto front has a specific structure where
            x_1 determines the position on the front. We sample x_1 uniformly
            and set other variables to values that tend to produce near-front points.
            """
            x = np.zeros(self.dim)
            # Sample x_1 uniformly - this determines position on the front
            x[0] = lo[0] + (hi[0] - lo[0]) * rng.random()
            # For remaining variables, use values that help maintain the front structure
            # In WFG4, the transformation involves specific functions of x_2...x_n
            # We sample them with a bias toward values that produce valid front points
            for i in range(1, self.dim):
                # Use a mixture: 50% uniform, 50% biased toward mid-range values
                if rng.random() < 0.5:
                    x[i] = lo[i] + (hi[i] - lo[i]) * rng.random()
                else:
                    # Bias toward values that are more likely to be on or near the front
                    mid = (lo[i] + hi[i]) / 2
                    spread = (hi[i] - lo[i]) / 4
                    x[i] = mid + spread * rng.standard_normal()
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Adaptive step size: start large, decrease over time
        base_step = 0.15
        min_step = 0.01

        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Adaptive step size based on progress
            progress = used / self.budget
            step_size = base_step * (1.0 - progress) + min_step * progress

            # Generate multiple children per iteration for better exploration
            n_children = min(3, self.budget - used)

            for _ in range(n_children):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # With 30% probability, use WFG4-biased sampling
                # With 70% probability, use standard evolutionary step
                if rng.random() < 0.3:
                    child = wfg4_biased_sample(rng)
                else:
                    parent = select_parent(F, X, rng)
                    child = parent + step_size * range_ * rng.standard_normal(self.dim)
                    child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Record point
                all_X.append(child)
                all_F.append(fc)

                # Replace worst solution in population using dominance
                worst_idx = None
                for i in range(len(F)):
                    if np.all(fc <= F[i]) and np.any(fc < F[i]):
                        worst_idx = i
                        break
                if worst_idx is None:
                    # If no dominated solution, replace the one with highest sum
                    worst_idx = int(np.argmax(np.sum(F, axis=1)))

                X[worst_idx] = child
                F[worst_idx] = fc

            # Periodically recompute and prune to maintain quality
            if used % 10 == 0:
                nd_mask = vectorized_non_dominated(F)
                if np.sum(nd_mask) < len(F) // 2:
                    # Keep non-dominated + some random dominated for diversity
                    nd_indices = np.where(nd_mask)[0]
                    dom_indices = np.where(~nd_mask)[0]
                    if len(dom_indices) > 0:
                        keep_dom = rng.choice(dom_indices, size=min(5, len(dom_indices)), replace=False)
                    else:
                        keep_dom = np.array([], dtype=int)
                    keep = np.concatenate([nd_indices, keep_dom])
                    keep = np.unique(keep)
                    if len(keep) < len(F):
                        X = X[keep]
                        F = F[keep]

        # Final polish: use remaining budget for small-step refinement
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline - 5:
            # Find best point and do local refinement
            nd_mask = vectorized_non_dominated(F)
            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) > 0:
                # Pick a few non-dominated points for local search
                n_polish = min(remaining, 10)
                polish_points = rng.choice(nd_indices, size=min(n_polish, len(nd_indices)), replace=False)
                for pi in polish_points:
                    if used >= self.budget or time.time() > deadline:
                        break
                    center = X[pi]
                    # Small step local search
                    for _ in range(min(3, self.budget - used)):
                        if used >= self.budget or time.time() > deadline:
                            break
                        child = center + 0.02 * range_ * rng.standard_normal(self.dim)
                        child = np.clip(child, lo, hi)
                        fc = problem(child)
                        used += 1
                        all_X.append(child)
                        all_F.append(fc)

        return all_X
# EVOLVE-BLOCK-END
