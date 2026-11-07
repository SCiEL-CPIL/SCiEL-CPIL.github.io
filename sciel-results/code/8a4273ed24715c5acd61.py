"""Deterministic LHS-based multi-objective optimizer for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    """Latin Hypercube Sampling + local refinement for WFG9 (3 objectives, 12 dims)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        n = self.budget
        dim = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Phase 1: Generate LHS samples
        # Use a single LHS sample for better stratification
        n_lhs = 150
        sampler = qmc.LatinHypercube(d=dim, seed=42)
        samples_unit = sampler.random(n=n_lhs)
        X_lhs = lo + (hi - lo) * samples_unit

        # Evaluate LHS points
        F_lhs = np.array([problem(x) for x in X_lhs])

        # Phase 2: Local refinement around promising points
        # Select top candidates based on non-dominated ranking
        # Use a simple dominance-based selection
        n_non_dominated = self._non_dominated_set(F_lhs)
        
        # Pick diverse points from the non-dominated set
        # Use a greedy farthest-point sampling in objective space
        n_refine = 30
        selected_indices = self._diverse_selection(F_lhs, n_non_dominated, n_refine)

        # For each selected point, do a local search
        X_refined = []
        F_refined = []
        
        for idx in selected_indices:
            x0 = X_lhs[idx]
            # Use Nelder-Mead to minimize the hypervolume contribution
            # We'll minimize the maximum objective as a proxy
            def obj_func(x):
                fx = problem(x)
                # Penalize points that are outside the reference point
                ref = np.array([3.0, 5.0, 7.0])
                if np.any(fx >= ref):
                    return np.sum(fx - ref) + 100.0
                return np.max(fx)
            
            res = minimize(obj_func, x0, method='Nelder-Mead', 
                          options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < np.max(F_lhs[idx]):
                X_refined.append(res.x)
                F_refined.append(problem(res.x))

        # Combine all points
        X_all = np.vstack([X_lhs, np.array(X_refined)]) if X_refined else X_lhs
        F_all = np.array([problem(x) for x in X_all])

        return X_all

    def _non_dominated_set(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        return np.where(~is_dominated)[0]

    def _diverse_selection(self, F, indices, k):
        """Select k diverse points from the given indices using farthest-point sampling."""
        if len(indices) <= k:
            return indices.tolist()
        
        # Normalize F for distance calculation
        F_sub = F[indices]
        F_min = F_sub.min(axis=0)
        F_max = F_sub.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1.0
        F_norm = (F_sub - F_min) / F_range

        # Start with the first point
        selected = [0]
        remaining = list(range(1, len(indices)))
        
        for _ in range(k - 1):
            if not remaining:
                break
            # Find the point farthest from all selected points
            max_dist = -1
            best_idx = remaining[0]
            for idx in remaining:
                min_dist_to_selected = min(np.linalg.norm(F_norm[idx] - F_norm[s]) for s in selected)
                if min_dist_to_selected > max_dist:
                    max_dist = min_dist_to_selected
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return [indices[i] for i in selected]
# EVOLVE-BLOCK-END
