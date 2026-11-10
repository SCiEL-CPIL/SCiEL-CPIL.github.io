"""Multi-objective optimizer for RE34 using LHS initialization with NSGA-II and HV-guided polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using LHS initialization, NSGA-II, and HV-guided local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 310.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Reference point for HV calculations (from task)
        ref_point = np.array([1705.0, 12.0, 0.3])

        def dominates(a, b):
            """Return True if a dominates b (a is better in all objectives, strictly in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_rank(F):
            """Return rank array and fronts (list of lists of indices)."""
            n = len(F)
            if n == 0:
                return np.array([], dtype=int), []
            rank = np.zeros(n, dtype=int)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [set() for _ in range(n)]

            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        dominated_set[i].add(j)
                        domination_count[j] += 1
                    elif dominates(F[j], F[i]):
                        dominated_set[j].add(i)
                        domination_count[i] += 1

            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            current_rank = 0
            for i in current_front:
                rank[i] = current_rank

            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            rank[j] = current_rank + 1
                            next_front.append(j)
                current_front = next_front
                current_rank += 1

            return rank, fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            m = F[indices].shape[1]

            for k in range(m):
                obj_vals = F[indices, k]
                min_val = np.min(obj_vals)
                max_val = np.max(obj_vals)
                if max_val - min_val < 1e-12:
                    continue
                sorted_idx = np.argsort(obj_vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / (max_val - min_val)

            return cd

        # Generate LHS design for initial population
        pop_size = 60

        def latin_hypercube_sample(n, dim, rng):
            """Generate n points using Latin Hypercube Sampling in [0,1]^dim."""
            points = np.zeros((n, dim))
            for d in range(dim):
                perm = rng.permutation(n)
                points[:, d] = (perm + rng.random(n)) / n
            return points

        # Generate LHS points and map to bounds
        lhs_unit = latin_hypercube_sample(pop_size, dim, rng)
        X = lo + (hi - lo) * lhs_unit

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Main loop
        while used < budget:
            if time.time() - start_time > deadline:
                break

            remaining_budget = budget - used

            # Check if we are in the polish phase (last 30% of budget)
            if remaining_budget <= budget * 0.30:
                # Local Polish Phase with HV-guided acceptance
                rank, fronts = non_dominated_rank(F)
                if not fronts:
                    break
                front_indices = fronts[0]

                if len(front_indices) == 0:
                    parent_idx = rng.integers(0, len(X))
                else:
                    # Pick a parent from the front
                    parent_idx = front_indices[rng.integers(0, len(front_indices))]

                # Generate a local neighbor with adaptive step size
                step_scale = 0.02 * (remaining_budget / (budget * 0.30)) + 0.003
                noise = rng.normal(0, step_scale, dim)
                trial = X[parent_idx] + noise
                trial = np.clip(trial, lo, hi)

                ft = problem(trial)
                used += 1

                # HV-guided acceptance: accept if it improves the front
                # Check if trial is non-dominated w.r.t. the current front
                is_dominated = False
                for fi in front_indices:
                    if dominates(F[fi], ft):
                        is_dominated = True
                        break

                if not is_dominated:
                    # Accept: replace the parent or add to population
                    # If it dominates the parent, replace parent
                    if dominates(ft, F[parent_idx]):
                        X[parent_idx] = trial
                        F[parent_idx] = ft
                    else:
                        # Replace the worst point in the population (highest rank, lowest crowding)
                        worst_idx = np.argmax(rank)
                        X[worst_idx] = trial
                        F[worst_idx] = ft
                else:
                    # Probabilistic acceptance to escape plateaus
                    if rng.random() < 0.2:
                        replace_idx = rng.integers(0, len(X))
                        X[replace_idx] = trial
                        F[replace_idx] = ft

            else:
                # Exploration Phase: NSGA-II with DE
                n_offspring = min(20, budget - used)
                offspring_X = []
                offspring_F = []

                # Compute current rank for adaptive F factor
                rank, fronts = non_dominated_rank(F)

                for _ in range(n_offspring):
                    if used >= budget:
                        break

                    # Select 3 parents for DE mutation
                    p1, p2, p3 = rng.integers(0, len(X), size=3)
                    while p2 == p3:
                        p3 = rng.integers(len(X))

                    # Adaptive F factor
                    base_F = 0.5
                    if len(fronts) > 1:
                        F_factor = base_F + 0.3 * (1.0 - rank[p1] / max(1, len(fronts) - 1))
                    else:
                        F_factor = base_F
                    F_factor = max(0.3, min(1.0, F_factor))
                    CR = 0.7 + rng.random() * 0.3

                    trial = X[p1].copy()
                    r = rng.random(dim)
                    for j in range(dim):
                        if r[j] < CR:
                            trial[j] = X[p1][j] + F_factor * (X[p2][j] - X[p3][j])
                        else:
                            trial[j] = X[p1][j]
                    trial = np.clip(trial, lo, hi)

                    ft = problem(trial)
                    used += 1
                    offspring_X.append(trial)
                    offspring_F.append(ft)

                if not offspring_X:
                    break

                offspring_X = np.array(offspring_X)
                offspring_F = np.array(offspring_F)

                # Combine parent and offspring
                combined_X = np.vstack([X, offspring_X])
                combined_F = np.vstack([F, offspring_F])

                # Environmental selection
                n_keep = len(X)
                rank, fronts = non_dominated_rank(combined_F)

                selected_indices = []
                for front in fronts:
                    if len(selected_indices) + len(front) <= n_keep:
                        selected_indices.extend(front)
                    else:
                        remaining = n_keep - len(selected_indices)
                        cd = crowding_distance(combined_F, front)
                        sorted_by_cd = np.argsort(-cd)
                        for i in range(remaining):
                            selected_indices.append(front[sorted_by_cd[i]])
                        break

                X = combined_X[selected_indices]
                F = combined_F[selected_indices]

        return X
# EVOLVE-BLOCK-END
