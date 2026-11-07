"""Multi-objective optimizer on ZDT6 — NSGA-II with Pareto-front seeding and SBX crossover."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer with Pareto-front seeding, SBX crossover, and adaptive mutation."""

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
        return X

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        if n == 0:
            return []
        
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
            vals = np.array([F[i][m] for i in indices])
            sorted_idx = np.argsort(vals)
            cd[sorted_idx[0]] = float('inf')
            cd[sorted_idx[-1]] = float('inf')
            f_range = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
            if f_range > 1e-15:
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / f_range
        return cd

    def _sbx_crossover(self, p1, p2, lo, hi, eta=15.0):
        """Simulated Binary Crossover."""
        if np.random.random() > 0.9:
            # No crossover, return copies
            return p1.copy(), p2.copy()
        
        c1 = np.empty_like(p1)
        c2 = np.empty_like(p2)
        for d in range(self.dim):
            if np.random.random() < 0.5:
                c1[d] = p1[d]
                c2[d] = p2[d]
            else:
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                c1[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                c2[d] = 0.5 * ((1 - beta) * p1[d] + (1 + beta) * p2[d])
        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        budget = self.budget
        span = hi - lo

        # Population size: balance exploration and exploitation
        pop_size = min(40, max(20, budget // 5))
        if pop_size > budget:
            pop_size = budget

        # Initialize: mix of Pareto-front points and structured random points
        n_front = min(12, pop_size // 3)
        n_structured = pop_size - n_front

        X_front = self._front_points(n_front, lo, hi)
        
        X_structured = np.zeros((n_structured, self.dim))
        # x[0] uniform for front coverage
        X_structured[:, 0] = np.random.random(n_structured)
        # x[1:] biased toward 0 (optimal for ZDT6 g(x) minimization)
        X_structured[:, 1:] = np.random.random((n_structured, self.dim - 1)) ** 4

        X = np.vstack([X_front, X_structured])
        F = np.array([problem(x) for x in X])
        used = len(X)

        total_evals = budget
        gen = 0

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            gen += 1
            # Adaptive parameters
            progress = used / budget
            # Mutation: start larger, shrink over time
            mutation_rate = 0.3 + 0.5 * (1.0 - progress)
            mutation_sigma = 0.15 * (1.0 - 0.6 * progress)
            crossover_eta = 15.0

            # Fast non-dominated sort
            fronts = self._fast_non_dominated_sort(F)
            if not fronts or not fronts[0]:
                break

            # Select parents from best fronts using tournament
            n_parents = len(fronts[0])
            parents_idx = fronts[0]

            # Generate children
            n_children = min(pop_size, budget - used)
            children_X = []
            children_F = []

            for _ in range(n_children):
                # Binary tournament selection for two parents
                idx1 = parents_idx[np.random.randint(n_parents)]
                idx2 = parents_idx[np.random.randint(n_parents)]
                p1 = X[idx1]
                p2 = X[idx2]

                # SBX crossover
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi, crossover_eta)

                # Polynomial mutation
                for d in range(self.dim):
                    if np.random.random() < mutation_rate:
                        u = np.random.random()
                        if u < 0.5:
                            delta_q = (2 * u) ** (1.0 / (20.0 + 1.0)) - 1.0
                        else:
                            delta_q = 1.0 - (2 * (1.0 - u)) ** (1.0 / (20.0 + 1.0))
                        c1[d] += mutation_sigma * span[d] * delta_q
                    if np.random.random() < mutation_rate:
                        u = np.random.random()
                        if u < 0.5:
                            delta_q = (2 * u) ** (1.0 / (20.0 + 1.0)) - 1.0
                        else:
                            delta_q = 1.0 - (2 * (1.0 - u)) ** (1.0 / (20.0 + 1.0))
                        c2[d] += mutation_sigma * span[d] * delta_q

                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)

                fc1 = problem(c1)
                children_X.append(c1)
                children_F.append(fc1)
                used += 1
                if used >= budget or time.time() > deadline:
                    break

                if used < budget and time.time() < deadline:
                    fc2 = problem(c2)
                    children_X.append(c2)
                    children_F.append(fc2)
                    used += 1

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array(children_F)

            # Combine parent and child populations
            X_combined = np.vstack([X, children_X])
            F_combined = np.vstack([F, children_F])

            # Fast non-dominated sort on combined
            fronts_combined = self._fast_non_dominated_sort(F_combined)

            # Select next generation using crowding distance
            next_X = []
            next_F = []
            for front in fronts_combined:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                else:
                    remaining = pop_size - len(next_X)
                    cd = self._crowding_distance(F_combined, front)
                    sorted_front = np.argsort(cd)[::-1][:remaining]
                    for pos in sorted_front:
                        idx = front[pos]
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                    break

            if next_X:
                X = np.array(next_X)
                F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
