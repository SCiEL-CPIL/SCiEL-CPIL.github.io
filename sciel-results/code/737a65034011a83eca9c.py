"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using weighted-sum scalarization with local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        used = 0
        all_points = []  # List of (x, f) tuples

        def eval_point(x):
            nonlocal used
            f = problem(x)
            used += 1
            return f

        # Phase 1: Weighted-sum scalarization with multiple weight vectors
        # Generate weight vectors on the simplex
        n_weights = 40
        weights = []
        for _ in range(2000):
            w = rng.random(self.n_obj)
            w = w / w.sum()
            if not any(np.allclose(w, ww) for ww in weights):
                weights.append(w)
                if len(weights) >= n_weights:
                    break

        # For each weight vector, do a local search from multiple starting points
        for w in weights:
            if used >= self.budget - 50 or time.time() > deadline:
                break

            # Multiple starting points for each weight
            n_starts = 3
            for s in range(n_starts):
                if used >= self.budget - 5 or time.time() > deadline:
                    break

                # Random start
                x = lo + rng.random(self.dim) * range_
                f = eval_point(x)
                all_points.append((x.copy(), f.copy()))

                # Local search: gradient-free using Nelder-Mead style
                # Simple approach: try perturbations in each dimension
                for iteration in range(10):
                    if used >= self.budget - 5 or time.time() > deadline:
                        break

                    best_x = x
                    best_f = f
                    best_scalar = np.dot(w, f)

                    # Try perturbations in each dimension
                    for d in range(self.dim):
                        for delta in [0.1, 0.05, 0.01, -0.1, -0.05, -0.01]:
                            if used >= self.budget - 5 or time.time() > deadline:
                                break

                            x_trial = best_x.copy()
                            x_trial[d] += delta * range_[d]
                            x_trial = np.clip(x_trial, lo, hi)

                            f_trial = eval_point(x_trial)
                            scalar_trial = np.dot(w, f_trial)

                            if scalar_trial < best_scalar:
                                best_scalar = scalar_trial
                                best_f = f_trial
                                best_x = x_trial.copy()

                    # Update current point
                    if np.array_equal(best_x, x):
                        break
                    x = best_x
                    f = best_f
                    all_points.append((x.copy(), f.copy()))

        # Phase 2: Use remaining budget for random exploration + local refinement
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            # Random points
            n_random = min(remaining // 2, 30)
            for _ in range(n_random):
                if used >= self.budget or time.time() > deadline:
                    break
                x = lo + rng.random(self.dim) * range_
                f = eval_point(x)
                all_points.append((x.copy(), f.copy()))

            # Local refinement around best points
            remaining2 = self.budget - used
            if remaining2 > 0 and all_points:
                # Sort by a proxy for quality (sum of objectives)
                all_points.sort(key=lambda p: np.sum(p[1]))
                n_refine = min(remaining2, 20)
                for i in range(min(n_refine, len(all_points))):
                    if used >= self.budget or time.time() > deadline:
                        break
                    x_base = all_points[i][0].copy()
                    for _ in range(3):
                        if used >= self.budget or time.time() > deadline:
                            break
                        x_trial = x_base + rng.standard_normal(self.dim) * 0.05 * range_
                        x_trial = np.clip(x_trial, lo, hi)
                        f_trial = eval_point(x_trial)
                        if np.sum(f_trial) < np.sum(all_points[i][1]):
                            all_points[i] = (x_trial.copy(), f_trial.copy())
                        else:
                            all_points.append((x_trial.copy(), f_trial.copy()))

        # Deduplicate: keep only non-dominated points
        if not all_points:
            return np.array([])

        F = np.array([p[1] for p in all_points])
        X = np.array([p[0] for p in all_points])

        # Find non-dominated points
        n = len(F)
        non_dominated = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(i)

        if non_dominated:
            return X[non_dominated]
        return X

# EVOLVE-BLOCK-END
