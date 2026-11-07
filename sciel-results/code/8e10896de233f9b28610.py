"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(100, self.budget // 2)
        if pop_size < 10:
            pop_size = 10

        # Initial random population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(f1, f2):
            """Return True if f1 dominates f2 (f1 <= f2 in all objectives and < in at least one)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts (each is a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            n_dominance = [0] * n  # number of solutions dominating i
            front = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_dominance[i] += 1
                if n_dominance[i] == 0:
                    front.append(i)

            fronts = []
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dominance[j] -= 1
                        if n_dominance[j] == 0:
                            next_front.append(j)
                front = next_front
            return fronts

        def crowding_distance(F, indices):
            """Calculate crowding distance for a list of indices."""
            if len(indices) <= 2:
                return [float('inf')] * len(indices)

            m = self.n_obj
            distances = np.zeros(len(indices))
            for k in range(m):
                obj_vals = F[indices, k]
                sorted_idx = np.argsort(obj_vals)
                distances[sorted_idx[0]] = float('inf')
                distances[sorted_idx[-1]] = float('inf')
                obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                if obj_range > 0:
                    for i in range(1, len(sorted_idx) - 1):
                        distances[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / obj_range
            return distances

        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = fast_non_dominated_sort(F)

            # Create offspring
            offspring_X = []
            offspring_F = []

            # Tournament selection and crossover/mutation
            for _ in range(pop_size):
                # Select two parents via tournament selection
                parents = []
                for _ in range(2):
                    # Tournament
                    candidates = rng.integers(pop_size, size=3)
                    # Rank by front
                    ranks = np.zeros(pop_size)
                    for rank, front in enumerate(fronts):
                        for idx in front:
                            ranks[idx] = rank
                    # Pick candidate with best rank
                    best_candidate = candidates[np.argmin(ranks[candidates])]
                    parents.append(best_candidate)

                # Simulated binary crossover
                p1 = X[parents[0]]
                p2 = X[parents[1]]
                if rng.random() < 0.9:  # crossover probability
                    eta = 20.0
                    beta = 1.0
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            if rng.random() < 0.5:
                                u = rng.random()
                                if u <= 0.5:
                                    alpha = 2.0 * u
                                    beta_val = (alpha + 1.0) ** (1.0 / (eta + 1.0))
                                else:
                                    alpha = 2.0 * (1.0 - u)
                                    beta_val = (alpha + 1.0) ** (1.0 / (eta + 1.0))
                                child1_val = 0.5 * ((1.0 + beta_val) * p1[d] + (1.0 - beta_val) * p2[d])
                                child2_val = 0.5 * ((1.0 - beta_val) * p1[d] + (1.0 + beta_val) * p2[d])
                            else:
                                child1_val = p1[d]
                                child2_val = p2[d]
                        else:
                            child1_val = p2[d]
                            child2_val = p1[d]
                        offspring_X.append(child1_val)
                        offspring_X.append(child2_val)
                else:
                    offspring_X.append(p1.copy())
                    offspring_X.append(p2.copy())

            # Mutation
            for i in range(len(offspring_X)):
                if rng.random() < 0.9:  # mutation probability
                    # Polynomial mutation
                    for d in range(self.dim):
                        if rng.random() < 1.0 / self.dim:
                            u = rng.random()
                            if u <= 0.5:
                                alpha = 2.0 * u
                                delta_q = (alpha + 1.0) ** (1.0 / (eta + 1.0)) - 1.0
                            else:
                                alpha = 2.0 * (1.0 - u)
                                delta_q = 1.0 - (alpha + 1.0) ** (1.0 / (eta + 1.0))
                            if p1 is not None and p2 is not None:
                                offspring_X[i][d] += delta_q * (hi[d] - lo[d])
                            else:
                                offspring_X[i][d] += delta_q * (hi[d] - lo[d])
                            offspring_X[i][d] = np.clip(offspring_X[i][d], lo[d], hi[d])

            # Convert to numpy array and evaluate
            offspring_X = np.array(offspring_X[:pop_size])
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sort on combined
            fronts = fast_non_dominated_sort(combined_F)

            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(next_X)
                    distances = crowding_distance(combined_F, front)
                    sorted_front = np.argsort(-distances)
                    for i in range(remaining):
                        idx = front[sorted_front[i]]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
