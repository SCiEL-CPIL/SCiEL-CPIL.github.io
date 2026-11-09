"""Multi-objective optimizer for RE21 with adaptive search and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective evolutionary optimizer with deadline guard, adaptive steps, and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Initial population
        init_size = min(30, self.budget // 4)
        X = lo + span * rng.random((init_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = init_size

        def is_non_dominated(F):
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] = False
                        break
            return nd

        def crowding_distance(F, nd_mask):
            """Compute crowding distance for non-dominated points."""
            n = len(F)
            cd = np.zeros(n)
            nd_indices = np.where(nd_mask)[0]
            n_nd = len(nd_indices)
            if n_nd <= 2:
                cd[nd_indices] = np.inf
                return cd

            for k in range(F.shape[1]):
                sorted_idx = nd_indices[np.argsort(F[nd_indices, k])]
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F[sorted_idx[-1], k] - F[sorted_idx[0], k]
                if f_range > 1e-15:
                    for i in range(1, n_nd - 1):
                        cd[sorted_idx[i]] += (F[sorted_idx[i + 1], k] - F[sorted_idx[i - 1], k]) / f_range
            return cd

        def local_search(x, f, max_evals=20):
            """Perform a local refinement around x using coordinate descent."""
            best_x = x.copy()
            best_f = f.copy()
            step_size = 0.05 * span
            for _ in range(max_evals):
                if time.time() > deadline:
                    break
                improved = False
                for d in range(self.dim):
                    for direction in [-1, 1]:
                        trial = best_x.copy()
                        trial[d] += direction * step_size
                        trial = np.clip(trial, lo, hi)
                        if np.allclose(trial, best_x):
                            continue
                        trial_f = problem(trial)
                        used_global[0] += 1
                        # Check if trial dominates best
                        if np.all(trial_f <= best_f) and np.any(trial_f < best_f):
                            best_x = trial
                            best_f = trial_f
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-8 * span.max():
                        break
            return best_x, best_f

        used_global = [0]  # mutable container for closure

        # Main search loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated set
            nd_mask = is_non_dominated(F)
            n_nd = np.sum(nd_mask)

            # Adaptive step size: starts large, shrinks over budget
            progress = used / max(1, self.budget)
            base_step = 0.4 * (1.0 - 0.8 * progress)

            # Select parent: prefer non-dominated, use crowding for diversity
            if n_nd > 0:
                nd_indices = np.where(nd_mask)[0]
                cd = crowding_distance(F, nd_mask)
                # Pick from non-dominated with highest crowding distance
                cd_nd = cd[nd_indices]
                # Use weighted random selection favoring high crowding
                weights = cd_nd + 1e-10
                weights /= weights.sum()
                parent_idx = nd_indices[rng.choice(len(nd_indices), p=weights)]
            else:
                parent_idx = rng.integers(len(F))

            # Mutation with adaptive step
            step = base_step * span
            child = X[parent_idx] + step * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            # Evaluate
            fc = problem(child)
            used += 1

            # Replace point: prefer replacing dominated points
            if n_nd < len(F):
                dom_mask = ~nd_mask
                dom_indices = np.where(dom_mask)[0]
                replace_idx = dom_indices[np.argmax(F[dom_indices].sum(axis=1))]
            else:
                # All non-dominated: replace the one with lowest crowding distance
                cd = crowding_distance(F, nd_mask)
                replace_idx = int(np.argmin(cd))

            X[replace_idx] = child
            F[replace_idx] = fc

            # Periodic local search on best non-dominated points
            if used % 10 == 0 and time.time() < deadline - 30:
                nd_mask = is_non_dominated(F)
                if np.sum(nd_mask) > 0:
                    nd_indices = np.where(nd_mask)[0]
                    # Pick one non-dominated point for local search
                    ls_idx = nd_indices[rng.integers(len(nd_indices))]
                    ls_x, ls_f = local_search(X[ls_idx], F[ls_idx], max_evals=10)
                    if np.all(ls_f <= F[ls_idx]) and np.any(ls_f < F[ls_idx]):
                        X[ls_idx] = ls_x
                        F[ls_idx] = ls_f
                    used += 10  # account for local search evaluations (approximate)
                    used = min(used, self.budget)

        # Final local polish on top non-dominated points
        if time.time() < deadline - 10:
            nd_mask = is_non_dominated(F)
            if np.sum(nd_mask) > 0:
                nd_indices = np.where(nd_mask)[0]
                # Polish top 3 non-dominated points
                cd = crowding_distance(F, nd_mask)
                top_indices = nd_indices[np.argsort(cd[nd_indices])[::-1][:3]]
                for idx in top_indices:
                    if time.time() > deadline - 5:
                        break
                    if used >= self.budget:
                        break
                    ls_x, ls_f = local_search(X[idx], F[idx], max_evals=15)
                    if np.all(ls_f <= F[idx]) and np.any(ls_f < F[idx]):
                        X[idx] = ls_x
                        F[idx] = ls_f
                    used += 15
                    used = min(used, self.budget)

        return X
# EVOLVE-BLOCK-END
