"""Multi-objective optimizer on WFG4 with NSGA-II + adaptive local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II with LHS initialization, tournament selection, and adaptive local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    @staticmethod
    def _fast_non_dominated_sort(F):
        """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    nd[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    nd[i] += 1
        front0 = [i for i in range(n) if nd[i] == 0]
        fronts = [front0]
        while front0:
            next_front = []
            for i in front0:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front0 = next_front
        return fronts

    @staticmethod
    def _crowding_distance(F, indices):
        """Compute crowding distance for points at given indices."""
        m = len(indices)
        if m <= 2:
            return np.full(m, np.inf)
        dist = np.zeros(m)
        n_obj = F.shape[1]
        for k in range(n_obj):
            vals = F[indices, k]
            v_min, v_max = vals.min(), vals.max()
            if v_max == v_min:
                continue
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for idx in range(1, m - 1):
                dist[order[idx]] += (vals[order[idx + 1]] - vals[order[idx - 1]]) / (v_max - v_min)
        return dist

    @staticmethod
    def _lhs_sample(rng, n, dim, lo, hi):
        """Latin Hypercube Sampling for better initial coverage."""
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            samples[:, d] = (perm + rng.random(n)) / n
        return lo + (hi - lo) * samples

    def _local_polish(self, problem, bounds, x, lo, hi, rng, max_evals=5):
        """Adaptive local search around a point."""
        dim = len(x)
        best_x = x.copy()
        best_f = np.array(problem(best_x))
        evals_used = 1
        step_size = 0.03

        while evals_used < max_evals:
            improved = False
            # Try random directions
            for _ in range(dim):
                if evals_used >= max_evals:
                    break
                direction = rng.standard_normal(dim)
                direction /= (np.linalg.norm(direction) + 1e-10)
                trial = best_x + step_size * (hi - lo) * direction
                trial = np.clip(trial, lo, hi)
                ft = problem(trial)
                evals_used += 1
                if np.all(ft <= best_f) and np.any(ft < best_f):
                    best_x = trial
                    best_f = ft
                    improved = True
                    break
            if improved:
                step_size *= 0.8
            else:
                step_size *= 0.5
                if step_size < 0.001:
                    break
        return best_x, best_f, evals_used

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget

        # Population size: use ~40 for good balance
        pop_size = min(40, max(20, budget // 5))
        used = 0

        # Initialize with LHS
        X = self._lhs_sample(rng, pop_size, dim, lo, hi)
        F = np.array([problem(x) for x in X])
        used += pop_size

        # Main NSGA-II loop
        while used < budget - 10:
            if time.time() > deadline:
                break

            # Create offspring population
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < pop_size:
                if time.time() > deadline:
                    break
                if used >= budget - 10:
                    break

                # Tournament selection (size 2)
                i1 = rng.integers(pop_size)
                i2 = rng.integers(pop_size)
                # Determine winner by non-domination and crowding
                # Simplified: pick the one with lower weighted sum for diversity
                w = rng.dirichlet(np.ones(n_obj))
                s1 = np.dot(F[i1], w)
                s2 = np.dot(F[i2], w)
                parent1 = i1 if s1 < s2 else i2

                i3 = rng.integers(pop_size)
                i4 = rng.integers(pop_size)
                s3 = np.dot(F[i3], w)
                s4 = np.dot(F[i4], w)
                parent2 = i3 if s3 < s4 else i4

                # SBX crossover
                beta = 20
                child = np.empty(dim)
                for d in range(dim):
                    if rng.random() < 0.5:
                        u = rng.random()
                        if u <= 1.0 / (beta + 1):
                            beta_q = ((2 * u) ** (1.0 / (beta + 1)))
                        elif u >= 1.0 - 1.0 / (beta + 1):
                            beta_q = ((2 * (1 - u)) ** (1.0 / (beta + 1)))
                        else:
                            beta_q = 1.0
                        child[d] = 0.5 * ((1 + beta_q) * X[parent1, d] + (1 - beta_q) * X[parent2, d])
                    else:
                        u = rng.random()
                        if u <= 1.0 / (beta + 1):
                            beta_q = ((2 * u) ** (1.0 / (beta + 1)))
                        elif u >= 1.0 - 1.0 / (beta + 1):
                            beta_q = ((2 * (1 - u)) ** (1.0 / (beta + 1)))
                        else:
                            beta_q = 1.0
                        child[d] = 0.5 * ((1 - beta_q) * X[parent1, d] + (1 + beta_q) * X[parent2, d])

                # Polynomial mutation
                eta = 20
                for d in range(dim):
                    if rng.random() < 0.9:
                        u = rng.random()
                        if u < 0.5:
                            xy = (1 + 2 * (child[d] - lo[d]) / (hi[d] - lo[d])) ** (eta + 1)
                            delta_q = (2 * (1 - 1 / xy)) ** (1.0 / (eta + 1))
                        else:
                            xy = (1 + 2 * (hi[d] - child[d]) / (hi[d] - lo[d])) ** (eta + 1)
                            delta_q = 1 - (2 * (1 - 1 / xy)) ** (1.0 / (eta + 1))
                        child[d] += delta_q * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)

                # Evaluate
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_F)

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(combined_F)

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front by crowding distance
                    remaining = pop_size - len(new_X)
                    cd = self._crowding_distance(combined_F, front)
                    selected = np.argsort(-cd)[:remaining]
                    for s in selected:
                        new_X.append(combined_X[front[s]])
                        new_F.append(combined_F[front[s]])
                    break

            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)

        # Final local polish on non-dominated points
        if used < budget and time.time() < deadline:
            fronts = self._fast_non_dominated_sort(F)
            front0 = fronts[0] if fronts else []

            # Polish a few diverse non-dominated points
            n_polish = min(len(front0), max(3, 10 // 5))
            if len(front0) > n_polish:
                cd = self._crowding_distance(F, front0)
                top_idx = np.argsort(-cd)[:n_polish]
                polish_idx = [front0[i] for i in top_idx]
            else:
                polish_idx = list(front0)

            for idx in polish_idx:
                if used >= budget or time.time() > deadline:
                    break
                remaining = budget - used
                if remaining < 3:
                    break
                max_evals = min(3, remaining)
                new_x, new_f, evals_used = self._local_polish(
                    problem, bounds, X[idx], lo, hi, rng, max_evals
                )
                used += evals_used - 1
                if np.all(new_f <= F[idx]) and np.any(new_f < F[idx]):
                    X[idx] = new_x
                    F[idx] = new_f

        return X
# EVOLVE-BLOCK-END
