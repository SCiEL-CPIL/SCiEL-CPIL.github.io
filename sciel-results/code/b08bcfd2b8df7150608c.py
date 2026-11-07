# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 using a hybrid approach:
    1) MOEA/D-style decomposition for structured exploration of the Pareto front.
    2) NSGA-II-style non-dominated sorting for diversity maintenance.
    3) Local refinement along the estimated Pareto front tangent."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size for MOEA/D
        N = 40

        # Generate weight vectors on the simplex for n_obj objectives
        weights = self._generate_weights(N, n_obj, rng)

        # Initialize population using Latin Hypercube Sampling
        X = self._lhs(rng, N, dim, lo, hi)
        F = np.array([problem(x) for x in X], dtype=float)
        used = N

        # Reference points for normalization
        F_min = F.min(axis=0)
        F_max = F.max(axis=0) + 1.0

        # Track all non-dominated points found so far
        all_nd_points = []
        all_nd_xs = []

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_front(F, indices):
            front = []
            for i in indices:
                is_dom = False
                for j in indices:
                    if j == i:
                        continue
                    if dominates(F[j], F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    front.append(i)
            return np.array(front)

        def crowding_distance(F, front_idx):
            if len(front_idx) <= 2:
                return np.ones(len(front_idx))
            cd = np.zeros(len(front_idx))
            for k in range(n_obj):
                vals = F[front_idx, k]
                mn, mx = vals.min(), vals.max()
                if mx - mn < 1e-12:
                    cd[:] = 1.0
                    continue
                order = np.argsort(vals)
                cd[order[0]] += 1e10
                cd[order[-1]] += 1e10
                for i in range(1, len(order) - 1):
                    cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (mx - mn)
            return cd

        # Main optimization loop
        while used < budget:
            if time.time() > deadline:
                break

            # Update reference points
            F_min = np.minimum(F_min, F.min(axis=0))
            F_max = np.maximum(F_max, F.max(axis=0))
            F_norm = (F - F_min) / (F_max - F_min + 1e-10)

            # For each weight vector, find the best individual and try to improve it
            for i in range(N):
                if used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                w = weights[i]

                # Tchebycheff aggregation: minimize max_j w_j * |f_j - z_j|
                # Simplified: minimize weighted sum of normalized objectives
                scores = np.sum(w * F_norm, axis=1)
                parent_idx = int(np.argmin(scores))

                # Crossover with a random individual
                other_idx = int(rng.integers(N))
                p1 = X[parent_idx]
                p2 = X[other_idx]

                # Simulated binary crossover
                child = np.zeros(dim)
                for d in range(dim):
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / 3.0)
                        child[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                    else:
                        beta = (2 * (1 - u)) ** (1.0 / 3.0)
                        child[d] = 0.5 * ((1 + beta) * p2[d] + (1 - beta) * p1[d])

                # Adaptive mutation
                gen = used / max(N, 1)
                mutation_scale = 0.2 * (1.0 + 0.5 * np.sin(gen * 0.1))
                child += mutation_scale * span * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Update reference points
                F_min = np.minimum(F_min, fc)
                F_max = np.maximum(F_max, fc)

                # Replace the parent if the child is better in the weighted sense
                child_score = np.sum(w * ((fc - F_min) / (F_max - F_min + 1e-10)))
                parent_score = scores[parent_idx]
                if child_score <= parent_score + 1e-10:
                    X[parent_idx] = child
                    F[parent_idx] = fc

                    # Check if the new point is non-dominated
                    is_nd = True
                    for j in range(len(F)):
                        if j == parent_idx:
                            continue
                        if dominates(F[j], fc):
                            is_nd = False
                            break
                    if is_nd:
                        # Remove dominated points from all_nd_points
                        new_nd = []
                        new_nd_xs = []
                        for k in range(len(all_nd_points)):
                            if not dominates(fc, all_nd_points[k]):
                                new_nd.append(all_nd_points[k])
                                new_nd_xs.append(all_nd_xs[k])
                        new_nd.append(fc)
                        new_nd_xs.append(child)
                        all_nd_points = new_nd
                        all_nd_xs = new_nd_xs

        # Phase 2: Local refinement along estimated Pareto front direction
        # Extract non-dominated points from the final population
        nd_front = non_dominated_front(F, list(range(len(F))))
        if len(nd_front) >= 2:
            # Estimate tangent direction in objective space
            F_nd = F[nd_front]
            F_centered = F_nd - F_nd.mean(axis=0)
            u, s, vt = np.linalg.svd(F_centered, full_matrices=False)
            tangent_obj = u[0]

            # For each ND point, try small perturbations in decision space
            for i in nd_front:
                if time.time() > deadline or used >= budget:
                    break
                best_x = X[i].copy()
                best_f = F[i].copy()
                # Try several random perturbations
                for _ in range(6):
                    if used >= budget:
                        break
                    pert = 0.02 * span * rng.standard_normal(dim)
                    cand = np.clip(best_x + pert, lo, hi)
                    cf = problem(cand)
                    used += 1
                    # Check if cand is better in at least one objective and not worse in others
                    if np.all(cf <= best_f + 1e-10) and np.any(cf < best_f - 1e-10):
                        best_x = cand
                        best_f = cf
                X[i] = best_x
                F[i] = best_f

        return X

    def _generate_weights(self, N, n_obj, rng):
        """Generate weight vectors on a simplex for n_obj objectives."""
        if n_obj == 3:
            # Generate points on the 2D simplex (triangle) for 3 objectives
            weights = []
            n_grid = int(np.ceil(np.sqrt(N)))
            for i in range(n_grid + 1):
                for j in range(n_grid + 1 - i):
                    k = n_grid - i - j
                    w = np.array([i / n_grid, j / n_grid, k / n_grid], dtype=float)
                    norm = np.sum(w)
                    if norm > 0:
                        w = w / norm
                    weights.append(w)
            # If we have too many, subsample
            if len(weights) > N:
                indices = rng.choice(len(weights), N, replace=False)
                weights = [weights[i] for i in indices]
            # If we have too few, add random points
            while len(weights) < N:
                w = rng.random(n_obj)
                w = w / np.sum(w)
                weights.append(w)
            return np.array(weights[:N])
        else:
            # General case: random weights on simplex
            weights = []
            for _ in range(N):
                w = rng.random(n_obj)
                w = w / np.sum(w)
                weights.append(w)
            return np.array(weights)

    def _lhs(self, rng, n, dim, lo, hi):
        """Latin Hypercube Sampling for better initial spread."""
        X = np.zeros((n, dim))
        for d in range(dim):
            # Generate n random intervals and pick one point from each
            perm = rng.permutation(n)
            for i in range(n):
                frac = (perm[i] + rng.random()) / n
                X[i, d] = lo[d] + frac * (hi[d] - lo[d])
        return X
# EVOLVE-BLOCK-END
