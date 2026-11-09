"""Multi-objective optimizer for RE21 using a fast, vectorized NSGA-II variant."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # Use global np.random (evaluator seeds it)
        # Population size
        pop_size = 50
        # Ensure we don't exceed budget
        if pop_size > self.budget:
            pop_size = self.budget

        # Initialize population
        X = lo + span * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Helper: fast non-dominated sorting for 2 objectives
        def non_dominated_sort(F):
            """Returns list of fronts (list of indices)."""
            n = len(F)
            if n == 0:
                return []
            # For 2 objectives, point i dominates j if F[i,0]<=F[j,0] and F[i,1]<=F[j,1] and at least one strict
            domination_count = np.zeros(n, dtype=int)
            dominated_by = [list() for _ in range(n)]

            for i in range(n):
                for j in range(i + 1, n):
                    if F[i, 0] <= F[j, 0] and F[i, 1] <= F[j, 1] and (F[i, 0] < F[j, 0] or F[i, 1] < F[j, 1]):
                        domination_count[j] += 1
                        dominated_by[i].append(j)
                    elif F[j, 0] <= F[i, 0] and F[j, 1] <= F[i, 1] and (F[j, 0] < F[i, 0] or F[j, 1] < F[i, 1]):
                        domination_count[i] += 1
                        dominated_by[j].append(i)

            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = set()
                for i in current_front:
                    for j in dominated_by[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.add(j)
                current_front = list(next_front)
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            F_sub = F[indices]
            for m in range(self.n_obj):
                # Sort by objective m
                order = np.argsort(F_sub[:, m])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                f_range = F_sub[order[-1], m] - F_sub[order[0], m]
                if f_range > 1e-15:
                    for k in range(1, n - 1):
                        dist[order[k]] += (F_sub[order[k + 1], m] - F_sub[order[k - 1], m]) / f_range
            return dist

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = non_dominated_sort(F)

            # Build mating pool: select parents from first fronts
            # Use tournament selection on ranks
            # Assign ranks
            rank = np.zeros(pop_size, dtype=int)
            for front_idx, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = front_idx

            # Create offspring
            offspring_X = []
            offspring_F = []
            num_offspring = min(pop_size, self.budget - used)

            for _ in range(num_offspring):
                if time.time() > deadline:
                    break
                # Tournament selection: pick 2 random, choose lower rank
                i1 = np.random.randint(pop_size)
                i2 = np.random.randint(pop_size)
                if rank[i1] <= rank[i2]:
                    p1_idx = i1
                else:
                    p1_idx = i2

                i3 = np.random.randint(pop_size)
                i4 = np.random.randint(pop_size)
                if rank[i3] <= rank[i4]:
                    p2_idx = i3
                else:
                    p2_idx = i4

                # SBX crossover
                if np.random.random() < 0.9:
                    # Simulated binary crossover
                    eta_c = 20.0
                    child1 = X[p1_idx].copy()
                    child2 = X[p2_idx].copy()
                    for d in range(self.dim):
                        if np.random.random() < 0.5:
                            x1 = X[p1_idx, d]
                            x2 = X[p2_idx, d]
                            if abs(x1 - x2) > 1e-14:
                                if np.random.random() <= 1.0 / (eta_c + 1.0):
                                    beta = 2.0 * np.random.random()
                                    alpha = (beta + 1.0) ** (eta_c + 1.0)
                                    beta_q = (1.0 / alpha) ** (1.0 / (eta_c + 1.0))
                                else:
                                    beta = 1.0 / (2.0 - 2.0 * np.random.random())
                                    alpha = (beta + 1.0) ** (eta_c + 1.0)
                                    beta_q = (1.0 / alpha) ** (1.0 / (eta_c + 1.0))
                                c1 = 0.5 * ((1.0 + beta_q) * x1 + (1.0 - beta_q) * x2)
                                c2 = 0.5 * ((1.0 - beta_q) * x1 + (1.0 + beta_q) * x2)
                                child1[d] = c1
                                child2[d] = c2
                    offspring_X.append(child1)
                    offspring_X.append(child2)
                else:
                    offspring_X.append(X[p1_idx].copy())
                    offspring_X.append(X[p2_idx].copy())

                # Mutation (polynomial)
                eta_m = 20.0
                for child in offspring_X[-2:]:
                    for d in range(self.dim):
                        if np.random.random() < 1.0 / self.dim:
                            x = child[d]
                            lo_d = lo[d]
                            hi_d = hi[d]
                            if hi_d - lo_d < 1e-14:
                                continue
                            if np.random.random() <= 2.0 / (eta_m + 1.0):
                                beta = 1.0 + 2.0 * (x - lo_d) / (hi_d - lo_d)
                                alpha = (beta + 1.0) ** (eta_m + 1.0)
                                delta_q = (1.0 / alpha) ** (1.0 / (eta_m + 1.0)) - 1.0
                            else:
                                beta = 1.0 + 2.0 * (hi_d - x) / (hi_d - lo_d)
                                alpha = (beta + 1.0) ** (eta_m + 1.0)
                                delta_q = 1.0 - (1.0 / alpha) ** (1.0 / (eta_m + 1.0))
                            child[d] = x + (hi_d - lo_d) * delta_q
                            child[d] = np.clip(child[d], lo_d, hi_d)

            # Evaluate offspring
            for ox in offspring_X:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                of = problem(ox)
                offspring_F.append(of)
                used += 1

            if not offspring_F:
                break

            offspring_F = np.array(offspring_F, dtype=np.float64)
            offspring_X = np.array(offspring_X[:len(offspring_F)], dtype=np.float64)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            combined_size = len(combined_F)

            # Non-dominated sorting on combined
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
                    # Need crowding distance to truncate
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(combined_F, front)
                    # Sort by crowding distance descending
                    order = np.argsort(-cd)
                    for k in range(min(remaining, len(order))):
                        idx = front[order[k]]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            if not next_X:
                break

            X = np.array(next_X, dtype=np.float64)
            F = np.array(next_F, dtype=np.float64)

        return X
# EVOLVE-BLOCK-END
