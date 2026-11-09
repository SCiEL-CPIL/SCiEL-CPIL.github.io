"""Multi-objective optimizer on RE37 using Sobol sampling + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using Sobol quasi-random sampling + local Gaussian refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, f1, f2):
        """Check if f1 dominates f2 (minimization)."""
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _is_non_dominated(self, f_new, F_list):
        """Check if f_new is non-dominated by any point in F_list."""
        for f in F_list:
            if self._dominates(f, f_new):
                return False
        return True

    def _sobol_sequence(self, n, dim, lo, hi):
        """Generate n quasi-random points using a simple Halton/Sobol-like sequence."""
        # Use Halton sequence for simplicity (deterministic, no external deps needed)
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        points = np.zeros((n, dim))
        for i in range(1, n + 1):
            for d in range(dim):
                # Use prime bases for each dimension
                bases = [2, 3, 5, 7, 11, 13, 17, 19]
                base = bases[d % len(bases)]
                points[i - 1, d] = lo[d] + halton(i, base) * (hi[d] - lo[d])
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        F_list = []
        X_list = []
        sigmas = [] # Per-individual step sizes
        evals_used = 0

        def add_point(x, f, sigma=None):
            nonlocal evals_used, F_list, X_list, sigmas
            if self._is_non_dominated(f, F_list):
                # Remove dominated points
                new_F_list = []
                new_X_list = []
                new_sigmas = []
                for i, f_old in enumerate(F_list):
                    if not self._dominates(f, f_old):
                        new_F_list.append(f_old)
                        new_X_list.append(X_list[i])
                        new_sigmas.append(sigmas[i] if sigmas else 0.1)
                
                F_list = new_F_list
                X_list = new_X_list
                sigmas = new_sigmas
                
                F_list.append(f)
                X_list.append(x)
                sigmas.append(sigma if sigma is not None else 0.1)

        # Phase 1: Sobol/Halton quasi-random sampling for uniform coverage
        n_init = 60
        X_sobol = self._sobol_sequence(n_init, self.dim, lo, hi)

        for x in X_sobol:
            f = problem(x)
            evals_used += 1
            add_point(x, f)

        if time.time() > deadline or evals_used >= self.budget:
            return np.array(F_list)

        # Phase 2: Adaptive Local Refinement
        # Each point in the Pareto front has its own sigma.
        # We cycle through the points. If a new point is found, we keep/increase sigma.
        # If not, we decrease sigma.
        
        # Initialize sigmas if not set (they should be set in add_point, but ensure)
        if not sigmas:
            sigmas = [0.1] * len(X_list)

        # Sort indices by "quality" (sum of objectives) to prioritize good points
        if len(X_list) > 0:
            F_arr = np.array(F_list)
            sums = np.sum(F_arr, axis=1)
            order = np.argsort(sums)
            X_sorted = [X_list[i] for i in order]
            S_sorted = [sigmas[i] for i in order]
        else:
            X_sorted = []
            S_sorted = []

        point_idx = 0
        total_points = len(X_sorted)

        while evals_used < self.budget and time.time() < deadline - 5.0:
            if total_points == 0:
                break

            if point_idx >= total_points:
                point_idx = 0

            x_base = X_sorted[point_idx]
            sigma = S_sorted[point_idx]
            
            # Generate a perturbation
            x_trial = x_base + np.random.normal(0, sigma, self.dim)
            x_trial = np.clip(x_trial, lo, hi)
            
            f_trial = problem(x_trial)
            evals_used += 1

            was_added = self._is_non_dominated(f_trial, F_list)
            if was_added:
                # Success: Keep sigma or slightly increase to explore more
                add_point(x_trial, f_trial, sigma=sigma * 1.1)
                # Re-sort and update structures
                if len(X_list) > 0:
                    F_arr = np.array(F_list)
                    sums = np.sum(F_arr, axis=1)
                    order = np.argsort(sums)
                    X_sorted = [X_list[i] for i in order]
                    S_sorted = [sigmas[i] for i in order]
                    total_points = len(X_sorted)
                    if point_idx >= total_points:
                        point_idx = 0
                # Note: add_point already updated F_list, X_list, sigmas
                # We need to update the local X_sorted/S_sorted to reflect the new state
                # The code above does this.
            else:
                # Failure: Decrease sigma to zoom in
                S_sorted[point_idx] = max(sigma * 0.8, 0.01)
                # Update the sigma in the main list as well
                # Find the index of this point in the main list?
                # It's easier to just update the main sigmas list if we can identify the point.
                # Since we are cycling through a sorted view, we should sync back.
                # However, for efficiency, we can just update the local S_sorted and 
                # periodically sync or assume the next successful add will re-sort.
                # But we need to update the actual `sigmas` list for the point at `point_idx` 
                # in the *current* sorted order.
                # Let's just update the local view for now, and re-sort on success.
                # If we don't re-sort on failure, the sigmas in `sigmas` (global) might be stale.
                # Let's update the global sigmas list too.
                # We need the index in the global list.
                # Since X_sorted is a permutation of X_list, we can find the index.
                # But doing this every step is expensive.
                # Alternative: Just update local S_sorted. The global `sigmas` will be updated 
                # when a new point is added (which triggers a re-sort and copy from global sigmas).
                # But if we never add a point, the global sigmas become stale.
                # Let's force a re-sync every 10 steps or so.
                if evals_used % 10 == 0:
                    F_arr = np.array(F_list)
                    sums = np.sum(F_arr, axis=1)
                    order = np.argsort(sums)
                    X_sorted = [X_list[i] for i in order]
                    S_sorted = [sigmas[i] for i in order]
                    total_points = len(X_sorted)
                    if point_idx >= total_points:
                        point_idx = 0

            point_idx += 1
            if point_idx >= total_points:
                point_idx = 0

        return np.array(F_list)
# EVOLVE-BLOCK-END
