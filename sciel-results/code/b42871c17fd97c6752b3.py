"""Optimized multi-objective search for ZDT3."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential-evolution based multi-objective optimizer for ZDT3."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Initial population: stratified sampling + some random
        pop_size = 50
        X = np.empty((pop_size, dim))
        F = np.empty((pop_size, 2))

        # First few points: spread across the space
        for i in range(pop_size):
            if i < 5:
                # Special points near the known ZDT3 Pareto front structure
                # ZDT3 front has peaks; x0 near 0, 0.0574, 0.1218, 0.2574, 0.3898, 0.7128, 0.9518
                special_x0 = [0.0, 0.0574, 0.1218, 0.2574, 0.3898]
                x = rng.random(dim)
                x[0] = special_x0[i]
            else:
                x = lo + (hi - lo) * rng.random(dim)
            X[i] = x
            F[i] = problem(x)

        used = pop_size

        def dominates(a, b):
            """Return True if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def get_nondominated_indices(F):
            """Return indices of non-dominated points."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if dominates(F[i], F[j]):
                        dominated[j] = True
            return np.where(~dominated)[0]

        def crowding_distance(F, nd_indices):
            """Compute crowding distance for non-dominated points."""
            m = len(nd_indices)
            if m <= 2:
                return np.full(m, np.inf)
            cd = np.zeros(m)
            for k in range(F.shape[1]):
                obj = F[nd_indices, k]
                sorted_idx = np.argsort(obj)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                obj_range = obj[sorted_idx[-1]] - obj[sorted_idx[0]]
                if obj_range > 0:
                    for i in range(1, m - 1):
                        cd[sorted_idx[i]] += (obj[sorted_idx[i + 1]] - obj[sorted_idx[i - 1]]) / obj_range
            return cd

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Get non-dominated set
            nd_idx = get_nondominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))

            # Select parents using tournament selection on non-dominated set
            if len(nd_idx) >= 2:
                cd = crowding_distance(F, nd_idx)
                # Pick 3 random candidates from non-dominated, choose best by crowding distance
                candidates = rng.choice(len(nd_idx), size=min(3, len(nd_idx)), replace=False)
                best_cand = candidates[np.argmax(cd[candidates])]
                p1 = X[nd_idx[best_cand]]
            else:
                p1 = X[nd_idx[0]]

            # Differential evolution: pick two more random parents
            p2_idx = rng.integers(len(F))
            p3_idx = rng.integers(len(F))
            p2 = X[p2_idx]
            p3 = X[p3_idx]

            # Mutation
            F_mut = 0.8
            CR = 0.9
            v = p1 + F_mut * (p2 - p3)
            v = np.clip(v, lo, hi)

            # Crossover (binomial)
            trial = p1.copy()
            j_rand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
            trial = np.clip(trial, lo, hi)

            # Evaluate
            f_trial = problem(trial)
            used += 1

            # Selection: replace if trial is better
            # Compare with p1
            if dominates(f_trial, F[nd_idx[best_cand] if len(nd_idx) >= 2 else 0]):
                X[nd_idx[best_cand] if len(nd_idx) >= 2 else 0] = trial
                F[nd_idx[best_cand] if len(nd_idx) >= 2 else 0] = f_trial
            else:
                # Replace the worst point in the population
                # Use a simple metric: sum of objectives, but prefer non-dominated
                # Find the point that is dominated by the most others
                worst_idx = 0
                worst_count = -1
                for i in range(len(F)):
                    count = sum(1 for j in range(len(F)) if j != i and dominates(F[j], F[i]))
                    if count > worst_count:
                        worst_count = count
                        worst_idx = i
                X[worst_idx] = trial
                F[worst_idx] = f_trial

        return X
# EVOLVE-BLOCK-END
