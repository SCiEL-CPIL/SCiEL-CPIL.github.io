"""Multi-objective optimizer with LHS init, NSGA-II, and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: LHS init, NSGA-II with adaptive mutation, deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Population size: balance exploration vs exploitation
        pop_size = min(30, max(10, budget // 6))
        # Ensure we can do at least one generation
        if pop_size * 2 > budget:
            pop_size = max(4, budget // 3)

        # Latin Hypercube initialization for better spread
        def lhs_sample(n, dim):
            samples = np.zeros((n, dim))
            for j in range(dim):
                perm = np.arange(n)
                np.random.shuffle(perm)
                samples[:, j] = (perm + np.random.random(n)) / n
            return samples

        # Use the seeded RNG (np.random.seed was called by evaluator)
        X = lo + (hi - lo) * lhs_sample(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)
            fronts = []
            front0 = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    front0.append(i)
            fronts.append(front0)
            while front0:
                next_front = []
                for i in front0:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if not next_front:
                    break
                fronts.append(next_front)
                front0 = next_front
            # Assign remaining (shouldn't happen but safety)
            assigned = set()
            for f in fronts:
                assigned.update(f)
            if len(assigned) < n:
                remaining = [i for i in range(n) if i not in assigned]
                if remaining:
                    fronts.append(remaining)
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            idx = np.array(indices)
            for m in range(self.n_obj):
                vals = F[idx, m]
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                vrange = vals[order[-1]] - vals[order[0]]
                if vrange > 1e-15:
                    for k in range(1, n - 1):
                        cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / vrange
            return cd

        while used < budget:
            if time.time() > deadline:
                break

            # Create offspring
            n_off = min(pop_size, budget - used)
            if n_off <= 0:
                break

            # Non-dominated sort for parent selection
            fronts = non_dominated_sort(F)

            # Select parents using tournament on fronts + crowding
            def select_parents(n_sel):
                parents = []
                for _ in range(n_sel):
                    # Tournament: pick 2 random, choose from better front, then crowding
                    i1 = np.random.randint(len(F))
                    i2 = np.random.randint(len(F))
                    # Determine front ranks
                    rank1, rank2 = 0, 0
                    for r, front in enumerate(fronts):
                        if i1 in front:
                            rank1 = r
                            break
                    for r, front in enumerate(fronts):
                        if i2 in front:
                            rank2 = r
                            break
                    if rank1 != rank2:
                        idx = i1 if rank1 < rank2 else i2
                    else:
                        # Same front, use crowding distance
                        front_idx = fronts[rank1]
                        cd = crowding_distance(F, front_idx)
                        pos1 = front_idx.index(i1) if i1 in front_idx else 0
                        pos2 = front_idx.index(i2) if i2 in front_idx else 0
                        idx = i1 if cd[pos1] > cd[pos2] else i2
                    parents.append(idx)
                return parents

            # Adaptive mutation scale: larger early, smaller late
            progress = used / max(budget, 1)
            scale = 0.15 * (1.0 - 0.7 * progress)

            offspring_X = []
            offspring_F = []
            for _ in range(n_off):
                p1, p2 = select_parents(2)
                # SBX-like crossover (blending)
                alpha = np.random.random()
                child = alpha * X[p1] + (1.0 - alpha) * X[p2]
                # Gaussian mutation
                child = child + scale * (hi - lo) * np.random.standard_normal(dim)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)
                if used >= budget or time.time() > deadline:
                    break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine and select next generation
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sort on combined
            all_fronts = non_dominated_sort(combined_F)

            # Select top pop_size
            new_X = []
            new_F = []
            for front in all_fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.append(combined_X[front])
                    new_F.append(combined_F[front])
                else:
                    # Need to select from this front based on crowding
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    order = np.argsort(cd)[::-1]  # descending crowding
                    selected = [front[k] for k in order[:remaining]]
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
