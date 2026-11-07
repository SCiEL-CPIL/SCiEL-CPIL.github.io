# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using a combination of:
    1. Uniform random initialization
    2. NSGA-II style evolutionary search with crowding distance
    3. Local refinement via Nelder-Mead on promising points
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use the global numpy random state (seeded by evaluator)
        # We'll use np.random directly to respect the evaluator's seeding

        # Track all evaluated points and their objectives
        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget:
                return None
            f = problem(x)
            f = np.asarray(f, dtype=float)
            all_X.append(np.asarray(x, dtype=float))
            all_F.append(f)
            used += 1
            return f

        # Phase 1: Random initialization
        pop_size = min(40, budget // 5)
        X_init = lo + (hi - lo) * np.random.random((pop_size, dim))
        for x in X_init:
            evaluate(x)

        # Phase 2: Evolutionary search (NSGA-II-like)
        # Maintain a population, select parents via non-dominated sorting + crowding
        # Use crossover and mutation

        def non_dominated_sort(F):
            """Return list of fronts (lists of indices) from non-dominated sorting."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of solutions dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of solutions dominating i
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
            current_front = [i for i in range(n) if n_dom[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            if len(indices) <= 2:
                return np.ones(len(indices))
            cd = np.zeros(len(indices))
            for m in range(n_obj):
                vals = F[indices, m]
                min_val = vals.min()
                max_val = vals.max()
                if max_val - min_val < 1e-15:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, len(order) - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (max_val - min_val)
            return cd

        # Population management
        pop_X = np.array(all_X[:pop_size])
        pop_F = np.array(all_F[:pop_size])

        # Evolutionary loop
        while used < budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(pop_F)

            # Select parents: take from first front(s)
            parent_indices = []
            for front in fronts:
                if len(parent_indices) + len(front) >= pop_size // 2:
                    # Take only what we need
                    remaining = pop_size // 2 - len(parent_indices)
                    if remaining > 0:
                        cd = crowding_distance(pop_F, front)
                        # Select top by crowding distance
                        order = np.argsort(-cd)
                        parent_indices.extend([front[i] for i in order[:remaining]])
                    break
                else:
                    parent_indices.extend(front)

            if len(parent_indices) < 2:
                parent_indices = list(range(len(pop_X)))

            # Generate children
            n_children = min(10, budget - used)
            for _ in range(n_children):
                if used >= budget or time.time() > deadline:
                    break

                # Select two parents
                p1_idx = parent_indices[np.random.randint(len(parent_indices))]
                p2_idx = parent_indices[np.random.randint(len(parent_indices))]
                p1 = pop_X[p1_idx]
                p2 = pop_X[p2_idx]

                # Simulated binary crossover
                child = np.zeros(dim)
                for d in range(dim):
                    u = np.random.random()
                    if u < 0.5:
                        beta = (2 * u) ** (1 / 20)
                        child[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                    else:
                        beta = (2 * (1 - u)) ** (1 / 20)
                        child[d] = 0.5 * ((1 + beta) * p2[d] + (1 - beta) * p1[d])

                # Mutation
                for d in range(dim):
                    if np.random.random() < 0.2:
                        child[d] += 0.1 * (hi[d] - lo[d]) * np.random.standard_normal()

                child = np.clip(child, lo, hi)
                fc = evaluate(child)
                if fc is None:
                    break

                # Replace worst point in population
                # Use a simple criterion: worst hypervolume contribution approximation
                # Approximate by sum of objectives (lower is better)
                worst_idx = np.argmax(np.sum(pop_F, axis=1))
                pop_X[worst_idx] = child
                pop_F[worst_idx] = fc

        # Phase 3: Local refinement of best points
        if used < budget and time.time() < deadline:
            # Find non-dominated points
            fronts = non_dominated_sort(np.array(all_F))
            if fronts:
                best_front = fronts[0]
                # Try local search on each point in the best front
                for idx in best_front:
                    if used >= budget or time.time() > deadline:
                        break
                    x0 = all_X[idx]
                    # Simple local perturbation
                    for _ in range(5):
                        if used >= budget or time.time() > deadline:
                            break
                        x_trial = x0 + 0.05 * (hi - lo) * np.random.standard_normal(dim)
                        x_trial = np.clip(x_trial, lo, hi)
                        ft = evaluate(x_trial)
                        if ft is not None and np.all(ft <= all_F[idx]):
                            x0 = x_trial

        return all_X
# EVOLVE-BLOCK-END
