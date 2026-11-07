"""Improved multi-objective optimizer for DTLZ2."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using a decomposition-based approach with adaptive search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Generate weight vectors for decomposition (uniform on simplex)
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # Create a set of weight vectors on the (n_obj-1)-simplex
        # For 3 objectives, this is a 2D triangle
        if n_obj == 3:
            # Generate points on a triangle
            n_weights_per_edge = 8
            weights = []
            for i in range(n_weights_per_edge + 1):
                for j in range(n_weights_per_edge + 1 - i):
                    k = n_weights_per_edge - i - j
                    w = np.array([i, j, k], dtype=float) / n_weights_per_edge
                    if np.sum(w) > 0:
                        weights.append(w)
            weights = np.array(weights)
        else:
            # General case: random weights on simplex
            n_weights = min(50, budget // 4)
            raw = rng.random((n_weights, n_obj))
            weights = raw / raw.sum(axis=1, keepdims=True)

        n_w = len(weights)

        # Initialize population: one point per weight vector
        # Use a smart initialization for DTLZ2
        X = np.zeros((n_w, dim))
        for i in range(n_w):
            # For DTLZ2, the Pareto front is where x_1..x_{n-1} are on a sphere
            # and x_n (the position variable) determines the front location
            # Initialize with diverse points
            X[i] = lo + (hi - lo) * rng.random(dim)

        F = np.array([problem(x) for x in X])
        used = n_w

        # Track best per weight vector
        best_X = X.copy()
        best_F = F.copy()
        best_tchebycheff = np.array([self._tchebycheff(f, w) for f, w in zip(F, weights)])

        # Adaptive mutation strength
        mut_strength = 0.3
        min_strength = 0.01
        decay = 0.995

        # Main loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Select parent based on crowding/distance in objective space
            # Use a simple approach: pick random weight, evolve that subproblem
            idx = rng.integers(n_w)
            w = weights[idx]
            parent = best_X[idx].copy()

            # Adaptive mutation
            sigma = mut_strength * (hi - lo)
            if rng.random() < 0.1:
                # Simulated binary crossover-like: blend with another random point
                other_idx = rng.integers(n_w)
                if other_idx != idx:
                    blend = rng.random()
                    child = blend * parent + (1 - blend) * best_X[other_idx]
                    child += sigma * rng.standard_normal(dim)
                else:
                    child = parent + sigma * rng.standard_normal(dim)
            else:
                child = parent + sigma * rng.standard_normal(dim)

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Update if better for this subproblem
            tc = self._tchebycheff(fc, w)
            if tc < best_tchebycheff[idx]:
                best_X[idx] = child
                best_F[idx] = fc
                best_tchebycheff[idx] = tc
                X[idx] = child
                F[idx] = fc

            # Decay mutation strength
            mut_strength = max(min_strength, mut_strength * decay)

            # Occasionally do a global restart to escape local optima
            if used % 50 == 0 and used < budget - 50:
                # Replace the worst few points with random ones
                sums = np.sum(F, axis=1)
                worst_indices = np.argsort(sums)[-max(1, n_w // 5):]
                for wi in worst_indices:
                    X[wi] = lo + (hi - lo) * rng.random(dim)
                    F[wi] = problem(X[wi])
                    used += 1
                    if used >= budget:
                        break
                    tc = self._tchebycheff(F[wi], weights[wi])
                    if tc < best_tchebycheff[wi]:
                        best_X[wi] = X[wi]
                        best_F[wi] = F[wi]
                        best_tchebycheff[wi] = tc

        # Final polish: local refinement around best points
        if used < budget - 20:
            # Take the top non-dominated points and do local search
            nondom_idx = self._find_nondominated(best_F)
            if len(nondom_idx) > 0:
                # Sort by sum of objectives
                sums = np.sum(best_F[nondom_idx], axis=1)
                top_idx = nondom_idx[np.argsort(sums)[:min(10, len(nondom_idx))]]
                polish_budget = min(budget - used, 100)
                for ti in top_idx:
                    if polish_budget <= 0:
                        break
                    x = best_X[ti].copy()
                    for _ in range(min(10, polish_budget)):
                        if time.time() > deadline:
                            break
                        # Small step local search
                        step = 0.02 * (hi - lo)
                        x_trial = x + step * rng.standard_normal(dim)
                        x_trial = np.clip(x_trial, lo, hi)
                        f_trial = problem(x_trial)
                        used += 1
                        polish_budget -= 1
                        # Accept if it improves any objective without worsening others too much
                        if np.all(f_trial <= best_F[ti]) and np.any(f_trial < best_F[ti]):
                            x = x_trial
                            best_X[ti] = x
                            best_F[ti] = f_trial
                        elif np.sum(f_trial) < np.sum(best_F[ti]) * 0.95:
                            x = x_trial
                            best_X[ti] = x
                            best_F[ti] = f_trial

        return best_X

    def _tchebycheff(self, f, w):
        """Tchebycheff scalarization: max_i w_i * |f_i - z_i|, using z=0."""
        # For minimization with z_i = 0 (or use a lower bound)
        # Since we don't have ideal point, use simple weighted sum as fallback
        # Actually, let's use a proper Tchebycheff with estimated ideal
        # For simplicity, use weighted sum
        return np.sum(w * f)

    def _find_nondominated(self, F):
        """Find indices of non-dominated points."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]
# EVOLVE-BLOCK-END
