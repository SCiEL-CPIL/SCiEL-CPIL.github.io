"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer combining simple ES with NSGA-II-style selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reserve budget for final polish
        polish_budget = 30
        main_budget = budget - polish_budget

        # Initial population
        pop_size = min(50, main_budget // 4)
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_mask(F):
            """Vectorized: return boolean mask of non-dominated points in F."""
            n = len(F)
            # F[i] is dominated if exists j: F[j] <= F[i] and F[j] != F[i]
            # For each i, check if any j dominates i
            # diff[j, i] = F[j] - F[i]; j dominates i if all(diff[j,i] <= 0) and any(diff[j,i] < 0)
            # Equivalently: for point i, it's dominated if exists j where F[j] <= F[i] elementwise and F[j] != F[i]
            # Vectorized: compute for all pairs
            # A[i,j] = True if i dominates j (i is better than j in all, strictly better in at least one)
            # F[:, None, :] - F[None, :, :] gives shape (n, n, n_obj)
            # For i to dominate j: all(F[i] <= F[j], axis=obj) and any(F[i] < F[j], axis=obj)
            diff = F[:, None, :] - F[None, :, :]  # diff[i,j] = F[i] - F[j]
            leq = np.all(diff <= 0, axis=2)  # leq[i,j]: F[i] <= F[j] elementwise
            lt = np.any(diff < 0, axis=2)    # lt[i,j]: F[i] < F[j] in at least one obj
            dominates = leq & lt  # dominates[i,j]: i dominates j
            # Point i is non-dominated if no j dominates i, i.e., no row j where dominates[j,i] is True
            # dominated_by[i] = any(dominates[:, i])
            dominated = np.any(dominates, axis=0)  # dominated[i] = True if some j dominates i
            return ~dominated

        def crowding_distance(F, mask):
            """Compute crowding distance for non-dominated points."""
            n = len(F)
            dist = np.zeros(n)
            if not np.any(mask) or np.sum(mask) <= 2:
                return dist
            idx = np.where(mask)[0]
            for m in range(n_obj):
                order = idx[np.argsort(F[idx, m])]
                if len(order) > 2:
                    dist[order[0]] = np.inf
                    dist[order[-1]] = np.inf
                    f_range = F[order[-1], m] - F[order[0], m]
                    if f_range > 1e-12:
                        for i in range(1, len(order) - 1):
                            dist[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / f_range
            return dist

        while used < main_budget:
            if time.time() > deadline:
                break

            # Find non-dominated front
            nd_mask = non_dominated_mask(F)
            nd_count = np.sum(nd_mask)

            if nd_count == 0:
                parents_idx = np.arange(len(F))
            else:
                parents_idx = np.where(nd_mask)[0]

            # Select parent via tournament on non-dominated + crowding
            if len(parents_idx) > 1:
                cd = crowding_distance(F, nd_mask)
                c1, c2 = np.random.choice(parents_idx, 2, replace=False)
                if cd[c1] >= cd[c2]:
                    parent = X[c1]
                else:
                    parent = X[c2]
            else:
                parent = X[parents_idx[0]]

            # Adaptive mutation: exponential decay for better exploration early, exploitation late
            progress = used / main_budget
            sigma = 0.3 * np.exp(-2.0 * progress) * (hi - lo)
            child = parent + sigma * np.random.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point
            dom_mask = ~nd_mask
            if np.any(dom_mask):
                dom_idx = np.where(dom_mask)[0]
                worst_idx = dom_idx[np.argmax(F[dom_idx].sum(axis=1))]
            else:
                cd = crowding_distance(F, nd_mask)
                worst_idx = np.argmin(cd)

            X[worst_idx] = child
            F[worst_idx] = fc

        # Final polish: refine non-dominated points with small perturbations
        if used < budget:
            nd_mask = non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]
            if len(nd_idx) > 0 and time.time() < deadline:
                for idx in nd_idx:
                    if used >= budget or time.time() > deadline:
                        break
                    # Try small perturbations around this point
                    best_x, best_f = X[idx], F[idx]
                    for _ in range(3):
                        if used >= budget or time.time() > deadline:
                            break
                        sigma_small = 0.02 * (hi - lo)
                        candidate = X[idx] + sigma_small * np.random.standard_normal(dim)
                        candidate = np.clip(candidate, lo, hi)
                        cf = problem(candidate)
                        used += 1
                        # Accept if it dominates or is non-dominated
                        if np.all(cf <= best_f) and np.any(cf < best_f):
                            best_x, best_f = candidate, cf
                    X[idx] = best_x
                    F[idx] = best_f

        return X
# EVOLVE-BLOCK-END
