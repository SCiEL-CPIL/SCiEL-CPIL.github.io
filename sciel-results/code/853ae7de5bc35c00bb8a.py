"""Multi-objective EGO optimizer for RE21: Pareto-based search with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Pareto-based multi-objective search with adaptive mutation and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _is_dominated(self, f, F):
        """Check if f is dominated by any point in F."""
        return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))

    def _pareto_tournament(self, F, n_candidates=2):
        """Select a parent using Pareto tournament selection."""
        n = len(F)
        if n == 0:
            return None
        if n == 1:
            return 0
        # Pick n_candidates random indices
        candidates = np.random.choice(n, size=min(n_candidates, n), replace=False)
        best = candidates[0]
        for c in candidates[1:]:
            # Check if c dominates best
            if np.all(F[c] <= F[best]) and np.any(F[c] < F[best]):
                best = c
            elif not (np.all(F[best] <= F[c]) and np.any(F[best] < F[c])):
                # If best doesn't dominate c, keep best (or pick randomly)
                pass
        return int(best)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng_size = hi - lo

        # Initial population
        pop_size = min(20, self.budget // 4)
        X = lo + rng_size * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Adaptive mutation scale
        base_scale = 0.3

        while used < self.budget:
            # Deadline check
            if time.time() > deadline:
                break

            # Pareto tournament selection
            parent_idx = self._pareto_tournament(F, n_candidates=2)
            if parent_idx is None:
                parent_idx = 0

            # Adaptive mutation: scale decreases as budget is consumed
            progress = used / self.budget
            scale = base_scale * (1.0 - 0.7 * progress)
            child = X[parent_idx] + scale * rng_size * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replacement: find the worst point that is dominated by child, or just replace worst by hypervolume contribution
            # Simple approach: replace the point that is dominated by child if any, else replace worst by sum
            dominated_mask = np.array([np.all(F[i] >= fc) and np.any(F[i] > fc) for i in range(len(F))])
            if np.any(dominated_mask):
                # Replace one of the dominated points
                dominated_indices = np.where(dominated_mask)[0]
                replace_idx = dominated_indices[rng.integers(len(dominated_indices))]
            else:
                # Replace the worst point by sum of objectives
                replace_idx = int(np.argmax(np.sum(F, axis=1)))

            X[replace_idx] = child
            F[replace_idx] = fc

        # Local refinement on best non-dominated points
        if time.time() < deadline - 30 and used < self.budget:
            # Find non-dominated points
            non_dom = []
            for i in range(len(F)):
                if not self._is_dominated(F[i], F):
                    non_dom.append(i)
            if non_dom:
                # Refine up to 5 best points with Nelder-Mead
                # Use a scalarized objective: sum of normalized objectives
                f_min = F.min(axis=0)
                f_range = F.max(axis=0) - f_min
                f_range[f_range == 0] = 1
                F_norm = (F - f_min) / f_range

                # Sort non-dominated by sum of normalized objectives
                non_dom_sorted = sorted(non_dom, key=lambda i: np.sum(F_norm[i]))
                n_refine = min(5, len(non_dom_sorted), (self.budget - used) // 2)
                for idx in non_dom_sorted[:n_refine]:
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break

                    def scalar_obj(x):
                        return np.sum(problem(x))

                    x0 = X[idx].copy()
                    try:
                        res = minimize(scalar_obj, x0, method='Nelder-Mead',
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': 10, 'xatol': 1e-6, 'fatol': 1e-6})
                        if res.fun < np.sum(F[idx]):
                            X[idx] = res.x
                            F[idx] = problem(res.x)
                            used += 1
                    except Exception:
                        pass

        return X
# EVOLVE-BLOCK-END
