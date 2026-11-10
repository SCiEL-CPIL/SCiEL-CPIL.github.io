"""Multi-objective optimizer for RE21 using NSGA-II with elitism."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer with elitism, non-dominated sorting, and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        pop_size = 50
        # Initial population
        X = lo + span * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def time_ok():
            return time.time() < deadline

        def non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination set
            n_dom = np.zeros(n, dtype=int)
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            return fronts[:-1]  # remove last empty front

        def crowding_distance(F, front_indices):
            """Assign crowding distances for points in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                obj_vals = F[front_indices, m]
                order = np.argsort(obj_vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                val_range = obj_vals[order[-1]] - obj_vals[order[0]]
                if val_range > 1e-30:
                    for i in range(1, n - 1):
                        cd[order[i]] += (obj_vals[order[i + 1]] - obj_vals[order[i - 1]]) / val_range
            return cd

        # Main loop
        while used < self.budget and time_ok():
            # Create offspring
            offspring_X = np.empty_like(X)
            offspring_F = np.empty_like(F)

            for i in range(pop_size):
                if not time_ok():
                    break

                # Tournament selection (size 2)
                i1, i2 = np.random.choice(pop_size, 2, replace=False)
                # Compare by front rank and crowding distance
                # We need ranks; compute them once per generation
                # For efficiency, precompute ranks
                # Actually let's compute ranks once per generation

                # This is inefficient; let's restructure
                pass

            # Restructure: compute fronts once per generation
            fronts = non_dominated_sort(F)
            rank = np.zeros(pop_size, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r

            # Compute crowding distance per front
            cd = np.zeros(pop_size)
            for front in fronts:
                cd[front] = crowding_distance(F, front)

            for i in range(pop_size):
                if not time_ok():
                    break

                # Tournament selection
                i1, i2 = np.random.choice(pop_size, 2, replace=False)
                if rank[i1] < rank[i2]:
                    p1 = i1
                elif rank[i2] < rank[i1]:
                    p1 = i2
                else:
                    p1 = i1 if cd[i1] > cd[i2] else i2

                i3, i4 = np.random.choice(pop_size, 2, replace=False)
                if rank[i3] < rank[i4]:
                    p2 = i3
                elif rank[i4] < rank[i3]:
                    p2 = i4
                else:
                    p2 = i3 if cd[i3] > cd[i4] else i4

                # Crossover (SBX-like with uniform mix)
                child = np.zeros(self.dim)
                for d in range(self.dim):
                    if np.random.random() < 0.5:
                        child[d] = X[p1, d]
                    else:
                        child[d] = X[p2, d]

                # Mutation
                if np.random.random() < 0.8:
                    child += 0.1 * span * np.random.standard_normal(self.dim)

                child = np.clip(child, lo, hi)
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1

            if not time_ok():
                break

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = 2 * pop_size

            # Non-dominated sort on combined
            fronts = non_dominated_sort(combined_F)

            # Select next population
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to sort by crowding distance
                    cd_front = crowding_distance(combined_F, front)
                    order = np.argsort(-cd_front)
                    needed = pop_size - len(next_X)
                    for pos in range(needed):
                        idx = front[order[pos]]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
