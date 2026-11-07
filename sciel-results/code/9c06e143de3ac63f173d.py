"""Multi-objective optimizer on WFG4 using NSGA-II with crowding distance."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        total_budget = self.budget

        # Initialize population using LHS for diversity
        def lhs_sample(n, dim, lo, hi, rng):
            """Latin Hypercube Sampling."""
            samples = np.zeros((n, dim))
            for d in range(dim):
                perm = rng.permutation(n)
                for i in range(n):
                    lower = lo[d] + i * (hi[d] - lo[d]) / n
                    upper = lo[d] + (i + 1) * (hi[d] - lo[d]) / n
                    samples[perm[i], d] = lower + rng.random() * (upper - lower)
            return samples

        pop_size = 30
        if total_budget < pop_size * 2:
            pop_size = max(10, total_budget // 3)

        # Initial population
        population = lhs_sample(pop_size, self.dim, lo, hi, rng)

        # Evaluate initial population
        evals = 0
        objectives = []
        for x in population:
            if evals >= total_budget:
                break
            f = problem(x)
            objectives.append(np.array(f))
            evals += 1

        objectives = np.array(objectives)
        # Trim population if we ran out of budget
        if len(objectives) < len(population):
            population = population[:len(objectives)]
            pop_size = len(population)

        # NSGA-II components
        def fast_non_dominated_sort(F):
            """Perform fast non-dominated sort on array of objective vectors."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_d = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1

            I = []
            for i in range(n):
                if n_d[i] == 0:
                    rank[i] = 0
                    I.append(i)

            k = 0
            while len(I) > 0:
                Q = []
                for i in I:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = k + 1
                            Q.append(j)
                I = Q
                k += 1

            fronts = []
            for r in range(k):
                front = [i for i in range(n) if rank[i] == r]
                if front:
                    fronts.append(front)
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for solutions in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            distances = np.zeros(n)
            m = self.n_obj

            for k in range(m):
                sorted_idx = sorted(range(n), key=lambda i: F[front_indices[i]][k])
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf

                f_min = F[front_indices[sorted_idx[0]]][k]
                f_max = F[front_indices[sorted_idx[-1]]][k]
                f_range = f_max - f_min

                if f_range > 0:
                    for i in range(1, n - 1):
                        prev = front_indices[sorted_idx[i - 1]]
                        next_ = front_indices[sorted_idx[i + 1]]
                        distances[sorted_idx[i]] += (F[prev][k] - F[next_][k]) / f_range

            return distances

        def tournament_select(pop, fronts, crowding, k=2):
            """Tournament selection based on rank and crowding distance."""
            indices = rng.integers(0, len(pop), k)
            best = indices[0]

            def get_rank(idx):
                for fi, front in enumerate(fronts):
                    if idx in front:
                        return fi
                return len(fronts)

            def get_cd(idx):
                fi = get_rank(idx)
                if fi < len(crowding):
                    front = fronts[fi]
                    pos = front.index(idx)
                    return crowding[fi][pos]
                return 0

            for idx in indices[1:]:
                r_best = get_rank(best)
                r_idx = get_rank(idx)
                if r_best < r_idx:
                    continue
                elif r_best > r_idx:
                    best = idx
                else:
                    if get_cd(idx) > get_cd(best):
                        best = idx
            return best

        def crossover(x1, x2):
            """Blend crossover with mutation."""
            alpha = rng.random(self.dim)
            child1 = alpha * x1 + (1 - alpha) * x2
            child2 = (1 - alpha) * x1 + alpha * x2

            sigma = 0.1 * (hi - lo)
            child1 = child1 + sigma * rng.standard_normal(self.dim)
            child2 = child2 + sigma * rng.standard_normal(self.dim)

            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            return child1, child2

        # Main NSGA-II loop
        while evals < total_budget:
            if time.time() > deadline - 5.0:
                break

            if len(population) < 2:
                break

            # Perform non-dominated sort
            fronts = fast_non_dominated_sort(objectives)

            # Compute crowding distances
            crowding = []
            for front in fronts:
                cd = crowding_distance(objectives, front)
                crowding.append(cd)

            # Generate offspring
            offspring = []
            offspring_obj = []

            n_offspring = min(pop_size, total_budget - evals)
            for _ in range(n_offspring):
                if evals >= total_budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                p1_idx = tournament_select(population, fronts, crowding)
                p2_idx = tournament_select(population, fronts, crowding)

                c1, c2 = crossover(population[p1_idx], population[p2_idx])

                if evals < total_budget:
                    f1 = problem(c1)
                    evals += 1
                    offspring.append(c1)
                    offspring_obj.append(np.array(f1))

                if evals < total_budget:
                    f2 = problem(c2)
                    evals += 1
                    offspring.append(c2)
                    offspring_obj.append(np.array(f2))

            if not offspring:
                break

            # Combine parent and offspring
            combined_pop = np.vstack([population, np.array(offspring)])
            combined_obj = np.vstack([objectives, np.array(offspring_obj)])

            # Non-dominated sort on combined
            combined_fronts = fast_non_dominated_sort(combined_obj)
            combined_crowding = []
            for front in combined_fronts:
                cd = crowding_distance(combined_obj, front)
                combined_crowding.append(cd)

            # Select next population
            new_pop = []
            new_obj = []
            for fi, front in enumerate(combined_fronts):
                if len(new_pop) + len(front) <= pop_size:
                    for idx in front:
                        new_pop.append(combined_pop[idx])
                        new_obj.append(combined_obj[idx])
                else:
                    needed = pop_size - len(new_pop)
                    cd = combined_crowding[fi]
                    sorted_front = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in range(min(needed, len(sorted_front))):
                        idx = front[sorted_front[i]]
                        new_pop.append(combined_pop[idx])
                        new_obj.append(combined_obj[idx])
                    break

            if not new_pop:
                break

            population = np.array(new_pop)
            objectives = np.array(new_obj)
            pop_size = len(population)

        # Return the non-dominated set
        if len(objectives) == 0:
            return np.array([])

        final_fronts = fast_non_dominated_sort(objectives)
        if final_fronts:
            best_front = final_fronts[0]
            return objectives[best_front]
        return objectives
# EVOLVE-BLOCK-END
