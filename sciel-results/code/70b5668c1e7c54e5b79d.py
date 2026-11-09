"""Multi-objective optimizer for RE37 using NSGA-II with adaptive mutation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, max(20, budget // 2))
        if pop_size > budget:
            pop_size = budget

        # Initialize population
        rng = np.random.default_rng(0)
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        evals = pop_size

        # Track all evaluated points for hypervolume
        all_F = [F.copy()]

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts (indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # Domination sets
            nD = np.zeros(n, dtype=int)  # Domination counts
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nD[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nD[i] += 1

            # First front
            for i in range(n):
                if nD[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nD[j] -= 1
                        if nD[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            fronts.pop()  # Remove last empty front
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.ones(n)
            
            dist = np.zeros(n)
            F_sub = F[indices]
            
            for m in range(n_obj):
                sorted_idx = np.argsort(F_sub[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
                if f_range > 1e-15:
                    for i in range(1, n - 1):
                        dist[sorted_idx[i]] += (F_sub[sorted_idx[i+1], m] - F_sub[sorted_idx[i-1], m]) / f_range
            return dist

        def tournament_select(fronts, F, indices, k=2):
            """Tournament selection based on non-domination rank and crowding distance."""
            # Compute ranks
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Compute crowding distances for each front
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cd_vals = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd[idx] = cd_vals[i]
            
            # Tournament
            candidates = rng.integers(len(F), size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best]:
                    best = c
                elif rank[c] == rank[best] and cd[c] > cd[best]:
                    best = c
            return best

        def crossover(x1, x2):
            """Simulated binary crossover."""
            if rng.random() < 0.9:  # Crossover probability
                eta = 20
                alpha = rng.random(dim)
                beta1 = 1 + 2 * alpha
                beta2 = 1 + 2 * (1 - alpha)
                u = rng.random(dim)
                
                cbeta1 = np.where(u <= 1/beta1, (2*u)**(1/(eta+1)), (2*(1-u))**(-1/(eta+1)))
                cbeta2 = np.where(u <= 1/beta2, (2*u)**(1/(eta+1)), (2*(1-u))**(-1/(eta+1)))
                
                child1 = 0.5 * ((1 + cbeta1) * x1 + (1 - cbeta1) * x2)
                child2 = 0.5 * ((1 + cbeta2) * x2 + (1 - cbeta2) * x1)
                return np.clip(child1, lo, hi), np.clip(child2, lo, hi)
            else:
                return x1.copy(), x2.copy()

        def mutate(x, sigma):
            """Polynomial mutation."""
            if rng.random() < 0.9:  # Mutation probability
                eta = 20
                u = rng.random(dim)
                delta1 = np.where(u < 0.5, (2*u)**(1/(eta+1)) - 1, 1 - (2*(1-u))**(-1/(eta+1)))
                x_mut = x + sigma * delta1
                return np.clip(x_mut, lo, hi)
            else:
                return x.copy()

        # Adaptive sigma
        sigma = 0.1

        while evals < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Evaluate current population
            fronts = fast_non_dominated_sort(F)

            # Selection and reproduction
            offspring = []
            offspring_F = []

            # Generate offspring until we have pop_size or run out of budget
            while len(offspring) < pop_size and evals + len(offspring) < budget:
                # Check deadline again
                if time.time() > deadline:
                    break

                # Tournament selection for two parents
                p1_idx = tournament_select(fronts, F, None)
                p2_idx = tournament_select(fronts, F, None)

                x1, x2 = X[p1_idx], X[p2_idx]
                c1, c2 = crossover(x1, x2)

                # Mutate
                c1 = mutate(c1, sigma)
                c2 = mutate(c2, sigma)

                offspring.append(c1)
                offspring.append(c2)

            if len(offspring) == 0:
                break

            # Evaluate offspring
            for child in offspring:
                if evals >= budget or time.time() > deadline:
                    break
                fc = problem(child)
                offspring_F.append(fc)
                evals += 1

            if len(offspring_F) == 0:
                break

            offspring_F = np.array(offspring_F)
            offspring = np.array(offspring[:len(offspring_F)])

            # Combine parents and offspring
            R_X = np.vstack([X, offspring])
            R_F = np.vstack([F, offspring_F])

            # Non-dominated sort on combined population
            R_fronts = fast_non_dominated_sort(R_F)

            # Compute crowding distances
            R_cd = np.zeros(len(R_F))
            for front in R_fronts:
                if len(front) > 0:
                    cd_vals = crowding_distance(R_F, front)
                    for i, idx in enumerate(front):
                        R_cd[idx] = cd_vals[i]

            # Select next generation
            new_X = []
            new_F = []
            for front in R_fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.append(R_X[front])
                    new_F.append(R_F[front])
                else:
                    # Need to select from this front
                    remaining = pop_size - len(new_X)
                    # Sort by crowding distance
                    front_cd = R_cd[front]
                    sorted_front = np.argsort(-front_cd)
                    selected = front[sorted_front[:remaining]]
                    new_X.append(R_X[selected])
                    new_F.append(R_F[selected])
                    break

            if len(new_X) == 0:
                break

            X = np.vstack(new_X)
            F = np.vstack(new_F)

            # Adapt sigma based on success rate
            # Simple adaptation: if we're making progress, keep sigma; else reduce
            # For simplicity, slowly decrease sigma
            sigma = max(0.01, sigma * 0.999)

        # Return the best non-dominated points
        return X
# EVOLVE-BLOCK-END
