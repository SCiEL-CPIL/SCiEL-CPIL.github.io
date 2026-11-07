"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective DE with non-dominated sorting and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 310.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            if np.all(np.isfinite(f)):
                all_F.append(f)
            used += 1
            return f

        # DE parameters
        F = 0.9  # mutation factor
        CR = 0.9  # crossover rate
        pop_size = 40

        # Initialize population uniformly in bounds
        X = []
        for i in range(pop_size):
            if time.time() > deadline:
                break
            x = rng.uniform(lo, hi)
            X.append(x)

        F_pop = np.array([evaluate(x) for x in X])

        # Non-dominated sorting helper
        def non_dominated_sort(F_mat):
            n = len(F_mat)
            ranks = np.zeros(n, dtype=int)
            fronts = []
            remaining = set(range(n))
            
            while remaining:
                front = []
                for i in remaining:
                    dominated = False
                    for j in remaining:
                        if i == j:
                            continue
                        if np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                            dominated = True
                            break
                    if not dominated:
                        front.append(i)
                for i in front:
                    ranks[i] = len(fronts)
                    remaining.discard(i)
                fronts.append(front)
            return ranks, fronts

        # Crowding distance helper
        def crowding_distance(F_mat, fronts):
            n = len(F_mat)
            crowding = np.zeros(n)
            for front in fronts:
                if len(front) <= 2:
                    for i in front:
                        crowding[i] = np.inf
                    continue
                for m in range(n_obj):
                    vals = F_mat[front, m]
                    order = np.argsort(vals)
                    crowding[front[order[0]]] = np.inf
                    crowding[front[order[-1]]] = np.inf
                    v_range = vals[order[-1]] - vals[order[0]]
                    if v_range > 0:
                        for k in range(1, len(front) - 1):
                            idx = front[order[k]]
                            crowding[idx] += (vals[order[k + 1]] - vals[order[k - 1]]) / v_range
            return crowding

        # Main DE loop
        while used < self.budget:
            if time.time() > deadline:
                break

            n = len(F_pop)
            ranks, fronts = non_dominated_sort(F_pop)
            crowding = crowding_distance(F_pop, fronts)

            # Generate offspring using DE/rand/1
            children_X = []
            children_F = []
            n_children = pop_size

            for _ in range(n_children):
                if time.time() > deadline or used >= self.budget:
                    break

                # Select three distinct random individuals
                idxs = rng.choice(n, size=3, replace=False)
                a = X[idxs[0]]
                b = X[idxs[1]]
                c = X[idxs[2]]

                # Mutation
                v = a + F * (b - c)

                # Crossover
                trial = np.zeros(dim)
                j_rand = rng.integers(0, dim)
                for d in range(dim):
                    if rng.random() < CR or d == j_rand:
                        trial[d] = v[d]
                    else:
                        trial[d] = X[rng.integers(0, n)][d]

                trial = np.clip(trial, lo, hi)
                f_trial = evaluate(trial)
                children_X.append(trial)
                children_F.append(f_trial)

            if not children_X:
                break

            # Combine parent and child populations
            combined_X = X + children_X
            combined_F = np.vstack([F_pop, np.array(children_F)])
            n_combined = len(combined_F)

            # Select best pop_size using non-dominated sorting and crowding distance
            ranks_c, fronts_c = non_dominated_sort(combined_F)
            crowding_c = crowding_distance(combined_F, fronts_c)

            selected = []
            for front in fronts_c:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    need = pop_size - len(selected)
                    front_sorted = sorted(front, key=lambda i: crowding_c[i], reverse=True)
                    selected.extend(front_sorted[:need])
                    break

            X = [combined_X[i] for i in selected]
            F_pop = combined_F[selected]

        return all_F
# EVOLVE-BLOCK-END
