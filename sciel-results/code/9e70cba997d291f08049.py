"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer tuned for DTLZ1 with small budget."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: keep it moderate for small budgets
        pop_size = min(30, max(10, budget // 5))

        # Use the global RNG (seeded by evaluator) for reproducibility
        # but also create a local RNG for internal operations
        rng = np.random.RandomState()

        # Initialize population with random points
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Record all evaluated points for final hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]

        def fast_non_dominated_sort(F):
            """Return list of fronts (lists of indices) and crowding distances."""
            n = len(F)
            S = [set() for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = [0] * n  # number of points dominating i
            fronts = [[]]

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n_dom[i] += 1

            for i in range(n):
                if n_dom[i] == 0:
                    fronts[0].append(i)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)

            # Remove last empty front
            if not fronts[-1]:
                fronts.pop()

            # Assign crowding distances
            crowding = np.zeros(n)
            for f_idx in range(len(fronts)):
                front = fronts[f_idx]
                m = len(front)
                if m <= 2:
                    for idx in front:
                        crowding[idx] = np.inf
                    continue
                for k in range(n_obj):
                    front_sorted = sorted(front, key=lambda idx: F[idx][k])
                    crowding[front_sorted[0]] = np.inf
                    crowding[front_sorted[-1]] = np.inf
                    f_range = F[front_sorted[-1]][k] - F[front_sorted[0]][k]
                    if f_range > 0:
                        for j in range(1, m - 1):
                            idx = front_sorted[j]
                            crowding[idx] += (F[front_sorted[j + 1]][k] - F[front_sorted[j - 1]][k]) / f_range

            return fronts, crowding

        def select_parents(F, crowding, fronts, n_select):
            """Select n_select parents using tournament selection."""
            selected = []
            while len(selected) < n_select:
                idx1 = rng.randint(len(F))
                idx2 = rng.randint(len(F))
                # Compare by front rank first, then crowding
                front1 = next(i for i, f in enumerate(fronts) if idx1 in f)
                front2 = next(i for i, f in enumerate(fronts) if idx2 in f)
                if front1 < front2:
                    selected.append(idx1)
                elif front2 < front1:
                    selected.append(idx2)
                else:
                    if crowding[idx1] >= crowding[idx2]:
                        selected.append(idx1)
                    else:
                        selected.append(idx2)
            return selected

        # Main loop: generate offspring, combine, select
        while used < budget:
            if time.time() > deadline:
                break

            # Determine how many offspring to generate in this generation
            remaining = budget - used
            n_offspring = min(pop_size, remaining)

            # Select parents
            fronts, crowding = fast_non_dominated_sort(F)
            parents = select_parents(F, crowding, fronts, n_offspring)

            # Generate offspring
            children_X = []
            children_F = []
            for p_idx in parents:
                # Simulated binary crossover-like mutation: perturb around parent
                child = X[p_idx].copy()
                # Adaptive step size: start larger, shrink over time
                progress = used / budget
                step = (0.2 * (1 - progress) + 0.02) * (hi - lo)
                child += step * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                children_X.append(child)
                children_F.append(problem(child))
                used += 1

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array(children_F)

            # Combine parent and offspring
            X_combined = np.vstack([X, children_X])
            F_combined = np.vstack([F, children_F])

            # Select next generation
            fronts, crowding = fast_non_dominated_sort(F_combined)
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                else:
                    # Sort by crowding and take remaining
                    remaining_slots = pop_size - len(next_X)
                    front_sorted = sorted(front, key=lambda idx: crowding[idx], reverse=True)
                    for idx in front_sorted[:remaining_slots]:
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

            # Record
            all_X.append(X.copy())
            all_F.append(F.copy())

        # Return the set of all non-dominated points found (for hypervolume)
        # The evaluator records all returned objective vectors, so we return
        # a representative set. Actually, the evaluator records every call to
        # problem() and uses all of them. So we just need to not exceed budget.
        # The return value is ignored per the contract.
        return X
# EVOLVE-BLOCK-END
