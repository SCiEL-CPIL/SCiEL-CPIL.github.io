"""Multi-objective optimizer for WFG9 using a combination of random sampling and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG9 using random sampling with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.perf_counter()
        deadline = t0 + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)

        all_F_list = []
        evals = 0

        def evaluate(x):
            nonlocal evals
            if evals >= budget:
                return None
            f = problem(x)
            evals += 1
            all_F_list.append(np.array(f, dtype=np.float64).copy())
            return f

        def is_dominated(f1, f2):
            """Check if f1 dominates f2 (f1 is better in all objectives, strictly better in at least one)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def fast_non_dominated_sort(F_arr):
            """Fast non-dominated sort. Returns list of fronts (each is a list of indices)."""
            n = len(F_arr)
            if n == 0:
                return []
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if is_dominated(F_arr[i], F_arr[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif is_dominated(F_arr[j], F_arr[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            front = [i for i in range(n) if n_dom[i] == 0]
            if front:
                fronts.append(front)

            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front = next_front

            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for a set of indices in F_arr."""
            n = len(indices)
            if n <= 2:
                return np.ones(n) * np.inf

            cd = np.zeros(n)
            idx = np.array(indices)
            F_sub = F_arr[idx]

            for m in range(n_obj):
                sorted_idx = np.argsort(F_sub[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf

                f_min = F_sub[sorted_idx[0], m]
                f_max = F_sub[sorted_idx[-1], m]
                if f_max - f_min < 1e-15:
                    continue

                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (F_sub[sorted_idx[k + 1], m] - F_sub[sorted_idx[k - 1], m]) / (f_max - f_min)

            return cd

        # Phase 1: Random initialization to get a diverse set of points
        n_init = min(30, budget // 5)
        if n_init < 10:
            n_init = 10

        X = np.empty((n_init, dim))
        F = np.empty((n_init, n_obj))

        for i in range(n_init):
            if evals >= budget:
                break
            x = lo + span * rng.random(dim)
            X[i] = x
            F[i] = evaluate(x)

        # Phase 2: Iterative refinement using local search around Pareto-optimal points
        # We'll maintain a set of non-dominated points and refine them
        n_refine = min(15, budget // 10)
        if n_refine < 5:
            n_refine = 5

        # Get current non-dominated points
        fronts = fast_non_dominated_sort(F)
        if fronts:
            pareto_indices = fronts[0]
        else:
            pareto_indices = list(range(n_init))

        # For each Pareto point, do local refinement
        for _ in range(n_refine):
            if evals >= budget or time.perf_counter() > deadline - 20:
                break

            # Select a Pareto point to refine
            idx = pareto_indices[rng.integers(0, len(pareto_indices))]
            x_center = X[idx].copy()

            # Try several perturbations
            n_tries = 3
            for _ in range(n_tries):
                if evals >= budget or time.perf_counter() > deadline - 20:
                    break

                # Gaussian perturbation with decreasing step size
                step_size = 0.05 + 0.05 * rng.random()
                x_new = x_center + step_size * span * rng.standard_normal(dim)
                x_new = np.clip(x_new, lo, hi)

                f_new = evaluate(x_new)
                if f_new is None:
                    break

                # Check if this is non-dominated
                dominated = False
                for i in range(len(F)):
                    if is_dominated(F[i], f_new):
                        dominated = True
                        break

                if not dominated:
                    # Add to population if we have room
                    if len(F) < 50:
                        F = np.vstack([F, f_new])
                        X = np.vstack([X, x_new])
                    else:
                        # Replace a dominated point if possible
                        for i in range(len(F)):
                            if is_dominated(f_new, F[i]):
                                F[i] = f_new
                                X[i] = x_new
                                break

        # Phase 3: NSGA-II-style evolutionary search for remaining budget
        pop_size = 40
        if evals + pop_size > budget:
            pop_size = max(10, budget - evals)

        # Initialize population from current non-dominated points
        fronts = fast_non_dominated_sort(F)
        if fronts:
            pareto_indices = fronts[0]
        else:
            pareto_indices = list(range(len(F)))

        if len(pareto_indices) >= pop_size:
            # Sample from Pareto front
            selected = rng.choice(pareto_indices, size=pop_size, replace=False)
        else:
            # Use all Pareto points and fill with random
            selected = list(pareto_indices)
            while len(selected) < pop_size and len(F) > 0:
                selected.append(rng.integers(0, len(F)))

        X_pop = np.empty((pop_size, dim))
        F_pop = np.empty((pop_size, n_obj))

        for i in range(pop_size):
            X_pop[i] = X[selected[i]]
            F_pop[i] = F[selected[i]]

        # NSGA-II loop
        while evals < budget:
            if time.perf_counter() > deadline - 10:
                break

            # Precompute ranks and crowding distances
            fronts = fast_non_dominated_sort(F_pop)
            rank = np.zeros(len(F_pop), dtype=int)
            cd_arr = np.zeros(len(F_pop))
            for front_id, front in enumerate(fronts):
                for i in front:
                    rank[i] = front_id
                if len(front) > 2:
                    cd_front = crowding_distance(F_pop, front)
                    for pos, idx in enumerate(front):
                        cd_arr[idx] = cd_front[pos]
                else:
                    for idx in front:
                        cd_arr[idx] = np.inf

            # Create offspring
            offspring_X = np.empty((pop_size, dim))
            offspring_F = np.empty((pop_size, n_obj))
            n_offspring = 0

            for i in range(pop_size):
                if evals >= budget or time.perf_counter() > deadline - 10:
                    break

                # Tournament selection
                def tournament_select():
                    candidates = rng.integers(0, len(F_pop), size=3)
                    best = candidates[0]
                    for c in candidates[1:]:
                        if rank[c] < rank[best] or (rank[c] == rank[best] and cd_arr[c] > cd_arr[best]):
                            best = c
                    return best

                p1 = tournament_select()
                p2 = tournament_select()

                # Blend crossover
                alpha = rng.random()
                child1 = (1 - alpha) * X_pop[p1] + alpha * X_pop[p2]
                child2 = (1 - alpha) * X_pop[p2] + alpha * X_pop[p1]

                # Mutation
                mutation_rate = 0.3
                for j in range(dim):
                    if rng.random() < mutation_rate:
                        child1[j] += 0.1 * span[j] * rng.standard_normal()
                    if rng.random() < mutation_rate:
                        child2[j] += 0.1 * span[j] * rng.standard_normal()

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                # Evaluate child1
                offspring_X[n_offspring] = child1
                offspring_F[n_offspring] = evaluate(child1)
                evals += 1
                n_offspring += 1

                if evals >= budget or time.perf_counter() > deadline - 10:
                    break

                # Evaluate child2
                if n_offspring < pop_size and evals < budget:
                    offspring_X[n_offspring] = child2
                    offspring_F[n_offspring] = evaluate(child2)
                    evals += 1
                    n_offspring += 1

            if n_offspring == 0:
                break

            # Combine parent and offspring
            combined_X = np.vstack([X_pop, offspring_X[:n_offspring]])
            combined_F = np.vstack([F_pop, offspring_F[:n_offspring]])
            combined_n = pop_size + n_offspring

            # Non-dominated sort on combined
            combined_fronts = fast_non_dominated_sort(combined_F)

            # Select next population
            new_X = np.empty((pop_size, dim))
            new_F = np.empty((pop_size, n_obj))
            count = 0

            for front in combined_fronts:
                if count + len(front) <= pop_size:
                    for idx in front:
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                else:
                    remaining = pop_size - count
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                    break

            X_pop = new_X
            F_pop = new_F

        # Return the final population (return value is ignored, but keep interface)
        return X_pop
# EVOLVE-BLOCK-END
