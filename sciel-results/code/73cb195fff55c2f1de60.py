import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        N = 12  # Population size
        F = 0.5  # Differential weight
        CR = 0.9  # Crossover rate

        # Initialize population
        X = lo + (hi - lo) * np.random.rand(N, self.dim)
        F_vals = np.array([problem(x) for x in X])
        used = N

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(points):
            n = len(points)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(points[i], points[j]):
                        S[i].append(j)
                    elif dominates(points[j], points[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

            k = 0
            while len(fronts[k]) > 0:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(points, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            f = points[front_indices]
            n_obj = self.n_obj
            for m in range(n_obj):
                min_val = f[:, m].min()
                max_val = f[:, m].max()
                if max_val - min_val == 0:
                    continue
                order = np.argsort(f[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[order[i]] += (f[order[i + 1], m] - f[order[i - 1], m]) / (max_val - min_val)
            return cd

        while used < self.budget:
            num_offspring = min(N, self.budget - used)
            U = np.zeros((num_offspring, self.dim))
            U_vals = np.zeros((num_offspring, self.n_obj))

            for i in range(num_offspring):
                r1, r2, r3 = np.random.choice(N, 3, replace=False)
                v = X[r1] + F * (X[r2] - X[r3])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() <= CR or j == j_rand:
                        U[i, j] = v[j]
                    else:
                        U[i, j] = X[i, j]
                U[i] = np.clip(U[i], lo, hi)
                U_vals[i] = problem(U[i])
                used += 1

            if used >= self.budget:
                break

            # Combine
            F_combined = np.vstack([F_vals, U_vals])
            X_combined = np.vstack([X, U])
            n_total = len(F_combined)

            # Non-dominated sort
            fronts = fast_non_dominated_sort(F_combined)

            # Select
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= N:
                    selected.extend(front)
                else:
                    need = N - len(selected)
                    cd = crowding_distance(F_combined, front)
                    order = np.argsort(-cd)
                    selected.extend(front[order[:need]])
                    break

            # If still not enough (shouldn't happen), fill
            if len(selected) < N:
                sel_set = set(selected)
                for i in range(n_total):
                    if i not in sel_set:
                        selected.append(i)
                        sel_set.add(i)
                        if len(selected) >= N:
                            break

            X = X_combined[selected]
            F_vals = F_combined[selected]

        return X
