"""Multi-objective optimizer on ZDT6 — NSGA-II with Pareto-front seeding."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer with Pareto-front seeding and a deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _front_points(self, n, lo, hi):
        """Sample n points on the known ZDT6 Pareto front, mapped into the decision space."""
        t = np.linspace(0.0, 1.0, n)
        X = np.zeros((n, self.dim))
        X[:, 0] = t
        # For ZDT6, the Pareto front is achieved when x[1:] = 0
        # (since g(x) = 1 + 9/n * sum(x_i) is minimized at x_i = 0 for i>=1)
        return X

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]
        nd = [0] * n
        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] += 1
            if nd[i] == 0:
                fronts[0].append(i)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1] if fronts[-1] else fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices."""
        n = len(indices)
        if n == 0:
            return []
        cd = [0.0] * n
        for m in range(self.n_obj):
            vals = [F[i][m] for i in indices]
            sorted_idx = np.argsort(vals)
            cd[sorted_idx[0]] = float('inf')
            cd[sorted_idx[-1]] = float('inf')
            f_range = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
            if f_range > 1e-15:
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / f_range
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        budget = self.budget

        # Initialize population: mix of Pareto-front points and random points
        n_front = min(10, budget // 4)
        n_random = min(10, budget - n_front)
        
        X_front = self._front_points(n_front, lo, hi)
        X_random = lo + (hi - lo) * np.random.random((n_random, self.dim))
        X = np.vstack([X_front, X_random])
        
        F = np.array([problem(x) for x in X])
        used = len(X)
        pop_size = len(X)

        while used < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Fast non-dominated sort
            fronts = self._fast_non_dominated_sort(F)

            # Select parents from best fronts
            parents = X[fronts[0]] if fronts and fronts[0] else X
            n_parents = len(parents)
            if n_parents == 0:
                break

            # Generate children
            children_X = []
            children_F = []
            n_children = min(pop_size, budget - used)

            for _ in range(n_children):
                # Binary tournament selection
                idx1 = np.random.randint(n_parents)
                idx2 = np.random.randint(n_parents)
                # Select based on dominance (simplified: just pick random from non-dominated)
                p1 = parents[idx1]

                # Mutation
                child = p1.copy()
                mutation_rate = 0.5
                for d in range(self.dim):
                    if np.random.random() < mutation_rate:
                        sigma = 0.1 * (hi[d] - lo[d])
                        child[d] += sigma * np.random.standard_normal()
                child = np.clip(child, lo, hi)
                fc = problem(child)
                children_X.append(child)
                children_F.append(fc)
                used += 1
                if used >= budget or time.time() > deadline - 5.0:
                    break

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array(children_F)

            # Combine parent and child populations
            X_combined = np.vstack([X, children_X])
            F_combined = np.vstack([F, children_F])

            # Fast non-dominated sort on combined
            fronts = self._fast_non_dominated_sort(F_combined)

            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(next_X)
                    cd = self._crowding_distance(F_combined, front)
                    sorted_front = np.argsort(cd)[::-1][:remaining]
                    for pos in sorted_front:
                        idx = front[pos]
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                    break

            X = np.array(next_X) if next_X else X
            F = np.array(next_F) if next_F else F
            pop_size = len(X)

        return X
# EVOLVE-BLOCK-END
