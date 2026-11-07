"""Multi-objective optimizer for RE37 using greedy distance-based search."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, dim):
        """Generate n Halton sequence points in [0,1)^dim."""
        points = np.zeros((n, dim))
        for i in range(1, n + 1):
            for d in range(dim):
                base = d + 2  # primes: 2, 3, 5, 7, ...
                f = 1.0
                result = 0.0
                j = i
                while j > 0:
                    f /= base
                    result += f * (j % base)
                    j //= base
                points[i - 1, d] = result
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Phase 1: Halton sequence initialization
        n_init = min(40, n)
        halton_pts = self._halton(n_init, dim)
        X_init = lo + halton_pts * span
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Check if we're already out of budget
        if used >= n:
            return X_init

        # Phase 2: Greedy farthest-point sampling in objective space
        # We want to maximize coverage of the Pareto front
        # At each step, pick the point that maximizes minimum distance to all existing objective vectors

        # Store all evaluated points and their objectives
        X_all = [X_init[i] for i in range(n_init)]
        F_all = [F_init[i] for i in range(n_init)]

        # For efficiency, keep F as a list of arrays
        # We'll use a candidate pool approach:
        # Generate many random candidates, evaluate the best one by distance criterion

        n_candidates = 50  # number of random candidates to consider per step
        rng = np.random.default_rng(42)

        while used < n:
            # Check time
            if time.time() > deadline:
                break

            # Generate candidates
            candidates = lo + span * rng.random((n_candidates, dim))
            # Also add some perturbations around existing points
            n_perturb = min(20, n_candidates)
            if n_perturb > 0:
                idxs = rng.integers(0, len(X_all), size=n_perturb)
                perturbations = X_all[idxs] + 0.15 * span * rng.standard_normal((n_perturb, dim))
                perturbations = np.clip(perturbations, lo, hi)
                candidates = np.vstack([candidates[:n_candidates - n_perturb], perturbations])

            # Evaluate all candidates
            F_cand = np.array([problem(x) for x in candidates])
            used += len(candidates)

            if used >= n:
                # Add all to our archive
                for i in range(len(candidates)):
                    X_all.append(candidates[i])
                    F_all.append(F_cand[i])
                break

            # Find the candidate that maximizes minimum distance to existing objectives
            # Only consider points that are not dominated (or at least competitive)
            # For simplicity, use all existing points for distance calculation

            # Compute distances from each candidate to all existing points
            # F_all is a list, convert to array for computation
            F_arr = np.array(F_all)  # shape (len(F_all), n_obj)
            # F_cand shape (n_candidates, n_obj)

            # Compute pairwise distances: for each candidate, min distance to any existing point
            # dist[i, j] = ||F_cand[i] - F_arr[j]||
            # We want argmax_i min_j dist[i, j]

            # Use broadcasting - but this could be large. Let's be careful.
            # F_cand: (n_cand, n_obj), F_arr: (n_existing, n_obj)
            # dist: (n_cand, n_existing)
            dists = np.sqrt(np.sum((F_cand[:, np.newaxis, :] - F_arr[np.newaxis, :, :]) ** 2, axis=2))
            min_dists = np.min(dists, axis=1)

            # Pick the candidate with the largest minimum distance
            best_idx = int(np.argmax(min_dists))
            X_all.append(candidates[best_idx])
            F_all.append(F_cand[best_idx])

        # Return the decision variables (interface requires returning something)
        return np.array(X_all)
