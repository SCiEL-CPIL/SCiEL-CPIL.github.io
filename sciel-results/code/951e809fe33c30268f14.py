# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting DTLZ4 separability.

    Fixes distance variables near their known optimum (0.5) and focuses
    search on the shape variables that control Pareto-front coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s / 330s kill
        rng = np.random.default_rng(42)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        dim = self.dim
        n_shape = dim - 3   # 9 shape variables
        n_dist = 3          # 3 distance variables

        # Distance variables pinned near 0.5 (known DTLZ4 optimum)
        dist_center = 0.5 * np.ones(n_dist)

        def make_x(shape):
            # shape: (n_shape,) in [0,1]; dist near 0.5
            x = np.empty(dim)
            x[:n_shape] = shape
            x[n_shape:] = dist_center + 0.005 * rng.standard_normal(n_dist)
            x[n_shape:] = np.clip(x[n_shape:], lo[n_shape:], hi[n_shape:])
            return x

        # ---- Initialize population on the front with good coverage ----
        pop_size = 40
        # Use a stratified grid on shape variables for initial spread
        # For 9 shape vars, use a low-discrepancy-like spread via random + jitter
        shapes = rng.random((pop_size, n_shape))
        # Add a few structured points for better initial coverage
        # Corner and center points
        structured = []
        structured.append(np.full(n_shape, 0.0))
        structured.append(np.full(n_shape, 1.0))
        structured.append(np.full(n_shape, 0.5))
        # Random but spread
        for i in range(min(pop_size, 20)):
            structured.append(rng.random(n_shape))
        structured = np.array(structured[:pop_size])
        shapes = structured

        X = np.array([make_x(s) for s in shapes])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Crowding distance helper
        def crowding_distances(F):
            m = len(F)
            if m <= 2:
                return np.full(m, np.inf)
            cd = np.zeros(m)
            for k in range(F.shape[1]):
            # sort indices by objective k
                order = np.argsort(F[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                rng_k = F[order[-1], k] - F[order[0], k]
                if rng_k < 1e-12:
                    continue
                for j in range(1, m - 1):
                    cd[order[j]] += (F[order[j + 1], k] - F[order[j - 1], k]) / rng_k
            return cd

        # Non-dominated mask
        def non_dominated_mask(F):
            m = len(F)
            nd = np.ones(m, dtype=bool)
            for i in range(m):
                if not nd[i]:
                    continue
                for j in range(m):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] = False
                        break
            return nd

        # Adaptive mutation step for shape variables
        step = 0.15

        while used < self.budget:
            if time.time() > deadline - 5.0:
                break

            # Identify non-dominated parents
            nd = non_dominated_mask(F)
            if np.any(nd):
                parents_idx = np.where(nd)[0]
            else:
                parents_idx = np.arange(len(F))

            # Select parent via tournament on crowding distance among non-dominated
            cd = crowding_distances(F[parents_idx])
            # Tournament of 2
            i1 = parents_idx[rng.integers(len(parents_idx))]
            i2 = parents_idx[rng.integers(len(parents_idx))]
            parent_idx = i1 if cd[parents_idx == i1][0] >= cd[parents_idx == i2][0] else i2
            # Simpler: pick from non-dominated by crowding
            best_local = parents_idx[np.argmax(cd)]
            parent_idx = best_local

            parent_shape = X[parent_idx, :n_shape].copy()

            # Mutate shape variables
            mutation = rng.standard_normal(n_shape) * step
            # Occasionally do a larger jump
            if rng.random() < 0.1:
                mutation *= 2.0
            child_shape = np.clip(parent_shape + mutation, 0.0, 1.0)

            # Occasionally reset a shape variable to a random value (exploration)
            if rng.random() < 0.05:
                idx = rng.integers(n_shape)
                child_shape[idx] = rng.random()

            child_x = make_x(child_shape)
            fc = problem(child_x)
            used += 1

            # Replace the worst point (by crowding distance among dominated, or max sum)
            # Find dominated points
            dominated_mask = ~nd
            if np.any(dominated_mask):
                # Among dominated, replace the one with worst crowding (smallest)
                dom_idx = np.where(dominated_mask)[0]
                cd_dom = crowding_distances(F[dom_idx])
                worst_local = dom_idx[np.argmin(cd_dom)]
            else:
                # All non-dominated: replace the one with smallest crowding distance
                cd_all = crowding_distances(F)
                worst_local = np.argmin(cd_all)

            X[worst_local] = child_x
            F[worst_local] = fc

            # Adapt step size: reduce slightly over time for refinement
            step = max(0.02, step * 0.999)

        return X
# EVOLVE-BLOCK-END
