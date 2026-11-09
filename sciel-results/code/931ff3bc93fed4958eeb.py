# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective optimizer with non-dominated sorting and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Population size
        N = 30
        budget_used = 0
        max_evals = self.budget

        # Initialize population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        budget_used += N

        def non_dominated_sort(F):
            """Return list of fronts (indices), each front is non-dominated."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []
            current_front = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[i] += 1  # wait, this is wrong. nd should count dominators
            # Let me redo this properly
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Does j dominate i?
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        nd[i] += 1
            current_front = [i for i in range(n) if nd[i] == 0]
            fronts.append(current_front)
            k = 0
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
                k += 1
            return fronts

        def crowding_distance(F, indices):
            """Assign crowding distance to points in indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            cd = np.zeros(len(indices))
            n_obj = F.shape[1]
            for m in range(n_obj):
                vals = F[indices, m]
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                rng_val = vals[order[-1]] - vals[order[0]]
                if rng_val > 1e-15:
                    for k in range(1, len(order) - 1):
                        cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / rng_val
            return cd

        def select_parents(F, X, n_sel):
            """Select n_sel parents using non-dominated sorting + crowding distance."""
            fronts = non_dominated_sort(F)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= n_sel:
                    selected.extend(front)
                else:
                    # Need to pick from this front
                    remaining = n_sel - len(selected)
                    cd = crowding_distance(F, front)
                    order = np.argsort(-cd)
                    selected.extend(front[order[:remaining]])
                    break
            return selected

        # Main DE loop
        CR = 0.9
        F_mut = 0.8

        while budget_used < max_evals:
            if time.time() > deadline - 5.0:
                break

            # Select parents
            parent_indices = select_parents(F, X, N)

            for i in range(N):
                if budget_used >= max_evals or time.time() > deadline - 5.0:
                    break

                # Find three distinct parents different from i
                candidates = [p for p in parent_indices if p != i]
                if len(candidates) < 3:
                    candidates = [p for p in range(N) if p != i]
                if len(candidates) < 3:
                    continue

                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

                # DE mutation
                v = X[r1] + F_mut * (X[r2] - X[r3])
                v = np.clip(v, lo, hi)

                # Crossover
                trial = X[i].copy()
                j_rand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                trial = np.clip(trial, lo, hi)

                # Evaluate
                f_trial = problem(trial)
                budget_used += 1

                # Selection
                if np.all(f_trial <= F[i]) and np.any(f_trial < F[i]):
                    X[i] = trial
                    F[i] = f_trial
                elif not (np.all(F[i] <= f_trial) and np.any(F[i] < f_trial)):
                    # Incomparable: keep the one with better crowding distance
                    # Simple heuristic: keep trial if it's in a better front
                    # For efficiency, just keep original if dominated, else random
                    if rng.random() < 0.5:
                        X[i] = trial
                        F[i] = f_trial

        return X
# EVOLVE-BLOCK-END
