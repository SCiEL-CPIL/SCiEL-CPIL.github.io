"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT3."""

    def __init__(self, dim, n_obj, budget):
        self.dim = dim
        self.n_obj = n_obj
        self.budget = budget

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Population size
        pop_size = min(200, max(20, budget // 2))
        # Keep remaining budget for evolution
        if pop_size > budget:
            pop_size = budget

        # Initialize population
        X = lo + range_ * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track best hypervolume-contributing points (non-dominated)
        # We'll maintain a set of non-dominated solutions

        def non_dominated_sort(F):
            """Return list of fronts, each front is a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            domination_count = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        domination_count[i] += 1

            current_front = [i for i in range(n) if domination_count[i] == 0]
            if current_front:
                fronts.append(current_front)

            idx = 0
            while idx < len(fronts) and idx + 1 < n:
                next_front = []
                for i in fronts[idx]:
                    for j in S[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                idx += 1

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distances for given indices in F."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            distances = np.zeros(n)
            for m in range(n_obj):
                obj_vals = F[indices, m]
                min_val = obj_vals.min()
                max_val = obj_vals.max()
                if max_val - min_val < 1e-10:
                    continue
                # Sort indices by this objective
                sorted_idx = np.argsort(obj_vals)
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    i = sorted_idx[k]
                    distances[i] += (obj_vals[sorted_idx[k + 1]] - obj_vals[sorted_idx[k - 1]]) / (max_val - min_val)

            return distances

        # Binary tournament selection
        def binary_tournament(F, fronts):
            """Select two individuals via binary tournament on front rank then crowding."""
            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i

            # Compute crowding distance
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cd[front] = crowding_distance(F, front)

            def better(i, j):
                if rank[i] != rank[j]:
                    return rank[i] < rank[j]
                return cd[i] > cd[j]

            a = np.random.randint(len(F))
            b = np.random.randint(len(F))
            if better(a, b):
                return a
            else:
                return b

        # SBX crossover
        def sbx_crossover(x1, x2, eta=20.0):
            """Simulated binary crossover."""
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x1)
            for i in range(len(x1)):
                if np.random.rand() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        child1[i] = x1[i]
                        child2[i] = x2[i]
                    else:
                        if np.random.rand() < 0.5:
                            beta = (2 * np.random.rand()) ** (1.0 / (eta + 1.0))
                        else:
                            beta = (1.0 / (2 * np.random.rand())) ** (1.0 / (eta + 1.0))
                        c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                        child1[i] = c1
                        child2[i] = c2
                else:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
            return child1, child2

        # Polynomial mutation
        def polynomial_mutation(x, eta=20.0):
            """Polynomial mutation."""
            child = x.copy()
            for i in range(len(x)):
                if np.random.rand() < 1.0 / dim:
                    if np.random.rand() < 0.5:
                        delta = (2 * np.random.rand()) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2 * np.random.rand()) ** (1.0 / (eta + 1.0))
                    child[i] += delta * range_[i]
                    child[i] = np.clip(child[i], lo[i], hi[i])
            return child

        # Main evolution loop
        while used < budget:
            if time.time() > deadline - 10.0:
                break

            # Compute fronts
            fronts = non_dominated_sort(F)

            # Create children
            n_children = min(pop_size, budget - used)
            children_X = []
            for _ in range(n_children // 2):
                p1 = binary_tournament(F, fronts)
                p2 = binary_tournament(F, fronts)
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                children_X.append(c1)
                children_X.append(c2)

            # If odd, add one more
            if len(children_X) < n_children:
                p1 = binary_tournament(F, fronts)
                c1 = polynomial_mutation(X[p1].copy())
                children_X.append(c1)

            # Evaluate children
            children_F = np.array([problem(x) for x in children_X])
            used += len(children_X)

            # Combine parents and children
            combined_X = np.vstack([X, children_X])
            combined_F = np.vstack([F, children_F])

            # Non-dominated sort
            fronts = non_dominated_sort(combined_F)

            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to sort by crowding distance
                    cd = crowding_distance(combined_F, front)
                    remaining = pop_size - len(new_X)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X

# EVOLVE-BLOCK-END
