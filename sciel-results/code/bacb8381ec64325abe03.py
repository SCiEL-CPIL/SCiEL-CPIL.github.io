"""Rewritten program for WFG4 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting WFG4 structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        span = hi - lo

        # WFG4 structure: x[0] determines shape (position on front),
        # x[1:] determines distance from front (bias).
        # Optimal bias variables are typically at midpoints or specific values.
        # For WFG4, the bias function g is minimized when x[1:] are at specific values.
        # Let's use a hybrid approach: dense sampling of shape variable + local search on bias.

        rng = np.random.default_rng(42)

        # Phase 1: Dense sampling of the shape variable (x[0])
        # With bias variables at midpoint (likely near-optimal for WFG4)
        n_shape = min(80, self.budget // 3)
        shape_vals = np.linspace(lo[0], hi[0], n_shape)

        # Try a few different bias settings
        bias_configs = [
            mid[1:],  # midpoint
            lo[1:],   # lower bound
            hi[1:],   # upper bound
            mid[1:] + 0.1 * span[1:],
            mid[1:] - 0.1 * span[1:],
        ]

        # Evaluate dense shape samples with best bias config
        best_f = []
        best_x = []

        # Start with midpoint bias
        x_template = np.empty(self.dim)
        x_template[1:] = mid[1:]

        for sv in shape_vals:
            if time.time() > deadline:
                break
            x_template[0] = sv
            f = problem(x_template)
            if np.all(f < [3.0, 5.0, 7.0]):  # strictly better than reference
                best_f.append(f)
                best_x.append(x_template.copy())

        # Convert to arrays
        if best_f:
            best_f = np.array(best_f)
            best_x = np.array(best_x)
        else:
            best_f = np.empty((0, self.n_obj))
            best_x = np.empty((0, self.dim))

        used = len(best_f)

        # Phase 2: Local search to improve bias variables for each shape point
        # Focus on points that are on or near the Pareto front
        if len(best_f) > 0:
            # Find non-dominated points
            nd_idx = self._find_non_dominated_indices(best_f)

            # For each non-dominated point, do local search on bias variables
            local_budget = min(self.budget - used - 20, 100)
            per_point = max(1, local_budget // max(1, len(nd_idx)))

            for idx in nd_idx:
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break

                x0 = best_x[idx].copy()
                f0 = best_f[idx].copy()

                # Local search on bias variables (x[1:])
                for _ in range(per_point):
                    if time.time() > deadline or used >= self.budget:
                        break

                    # Random perturbation of bias variables
                    step_size = 0.05 * span[1:]
                    x_new = x0.copy()
                    x_new[1:] += rng.standard_normal(self.dim - 1) * step_size
                    x_new = np.clip(x_new, lo, hi)

                    f_new = problem(x_new)
                    used += 1

                    # Accept if it dominates or is non-dominated
                    if np.all(f_new <= f0) and np.any(f_new < f0):
                        x0 = x_new
                        f0 = f_new
                    elif not self._dominates(f0, f_new) and not self._dominates(f_new, f0):
                        # Non-dominated, keep as alternative
                        pass

                # Update the point
                best_x[idx] = x0
                best_f[idx] = f0

        # Phase 3: Add some random exploration with remaining budget
        remaining = self.budget - used
        if remaining > 0:
            n_random = min(remaining, 30)
            for _ in range(n_random):
                if time.time() > deadline or used >= self.budget:
                    break
                x_rand = lo + span * rng.random(self.dim)
                f_rand = problem(x_rand)
                used += 1
                if np.all(f_rand < [3.0, 5.0, 7.0]):
                    best_f = np.vstack([best_f, f_rand]) if len(best_f) > 0 else f_rand.reshape(1, -1)
                    best_x = np.vstack([best_x, x_rand]) if len(best_x) > 0 else x_rand.reshape(1, -1)

        # Final: filter to only points strictly better than reference point
        ref = np.array([3.0, 5.0, 7.0])
        if len(best_f) > 0:
            mask = np.all(best_f < ref, axis=1)
            best_f = best_f[mask]
            best_x = best_x[mask]

        return best_x

    def _find_non_dominated_indices(self, F):
        """Find indices of non-dominated points."""
        n = len(F)
        if n == 0:
            return []
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            for j in range(n):
                if i == j or not nd[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] = False
                    break
        return np.where(nd)[0]

    def _dominates(self, a, b):
        """Check if a dominates b (a is better or equal in all, strictly better in at least one)."""
        return np.all(a <= b) and np.any(a < b)
# EVOLVE-BLOCK-END
