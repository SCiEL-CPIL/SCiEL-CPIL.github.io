"""Multi-objective optimizer for WFG7."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG7 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = 30
        max_evals = budget

        # Initial population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        evals_used = pop_size

        def fast_nondominated_sort(F):
            """Return list of fronts (indices), sorted by dominance rank."""
            n = len(F)
            S = [[] for _ in range(n)]
            dom_count = np.zeros(n, dtype=int)
            fronts = []
            current_front = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        dom_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dom_count[i] += 1

            for i in range(n):
                if dom_count[i] == 0:
                    current_front.append(i)

            idx = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        dom_count[j] -= 1
                        if dom_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
                idx += 1

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices in F."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            n = len(indices)
            dist = np.zeros(n)
            for m in range(n_obj):
                obj_vals = F[indices, m]
                min_v = obj_vals.min()
                max_v = obj_vals.max()
                if max_v - min_v < 1e-12:
                    continue
                order = np.argsort(obj_vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[order[k]] += (obj_vals[order[k + 1]] - obj_vals[order[k - 1]]) / (max_v - min_v)
            return dist

        # Main evolutionary loop
        while evals_used < max_evals:
            # Check deadline
            if time.time() > deadline:
                break

            # Sort into fronts
            fronts = fast_nondominated_sort(F)

            # Select parents: take from first fronts
            parents_pool = []
            for front in fronts:
                parents_pool.extend(front)
                if len(parents_pool) >= pop_size // 2:
                    break

            if not parents_pool:
                parents_pool = list(range(pop_size))

            # Generate children
            n_children = min(pop_size, max_evals - evals_used)
            if n_children <= 0:
                break

            children_X = []
            children_F = []

            for _ in range(n_children):
                # Pick two parents (tournament-like: random from pool)
                p1 = parents_pool[rng.integers(len(parents_pool))]
                p2 = parents_pool[rng.integers(len(parents_pool))]

                # SBX crossover
                if rng.random() < 0.9:
                    eta = 20.0
                    u = rng.random(dim)
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (2 * (1 - u)) ** (1.0 / (eta + 1))
                    child1 = 0.5 * ((1 + beta) * X[p1] + (1 - beta) * X[p2])
                    child2 = 0.5 * ((1 - beta) * X[p1] + (1 + beta) * X[p2])
                    # Mutate
                    for child in [child1, child2]:
                        if rng.random() < 0.5:
                            child += 0.1 * (hi - lo) * rng.standard_normal(dim)
                    children_X.append(np.clip(child1, lo, hi))
                    children_X.append(np.clip(child2, lo, hi))
                    if evals_used + 2 > max_evals:
                        break
                else:
                    # Just mutation
                    child = X[p1].copy()
                    child += 0.1 * (hi - lo) * rng.standard_normal(dim)
                    children_X.append(np.clip(child, lo, hi))

                if evals_used + len(children_X) > max_evals:
                    break

            # Evaluate children
            for cx in children_X:
                if evals_used >= max_evals:
                    break
                cf = problem(cx)
                children_F.append(cf)
                evals_used += 1

            if not children_F:
                break

            children_F = np.array(children_F, dtype=float)
            children_X = np.array(children_X, dtype=float)

            # Combine parents and children
            combined_X = np.vstack([X, children_X])
            combined_F = np.vstack([F, children_F])
            combined_n = len(combined_F)

            # Fast non-dominated sort on combined
            combined_fronts = fast_nondominated_sort(combined_F)

            # Select next population
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.append(combined_X[front])
                    new_F.append(combined_F[front])
                else:
                    # Need to pick from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    if remaining <= 0:
                        break
                    cd = crowding_distance(combined_F, front)
                    # Sort by crowding distance descending
                    order = np.argsort(-cd)
                    selected = front[order[:remaining]]
                    new_X.append(combined_X[selected])
                    new_F.append(combined_F[selected])
                    break

            if new_X:
                X = np.vstack(new_X)
                F = np.vstack(new_F)
            else:
                break

        return X
# EVOLVE-BLOCK-END
