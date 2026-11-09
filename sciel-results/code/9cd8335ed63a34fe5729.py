"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective Differential Evolution optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        # Population size
        # DE works well with populations of 5-10 * dim, but budget is tight (200).
        # Let's try pop_size = 40 to allow ~5 generations.
        pop_size = 40
        if self.budget < pop_size * 2:
            pop_size = max(10, self.budget // 2)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(fitnesses):
            """Return list of fronts, each front is a list of indices."""
            N = len(fitnesses)
            S = [[] for _ in range(N)]
            n_dom = [0] * N
            rank = [0] * N
            fronts = [[]]

            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    # i dominates j if i is better in all and strictly better in at least one
                    if np.all(fitnesses[i] <= fitnesses[j]) and np.any(fitnesses[i] < fitnesses[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(fitnesses[j] <= fitnesses[i]) and np.any(fitnesses[j] < fitnesses[i]):
                        n_dom[j] += 1

                if n_dom[i] == 0:
                    rank[i] = 0
                    fronts[0].append(i)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                i += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(front_indices, fitnesses):
            """Compute crowding distance for a front."""
            N = len(front_indices)
            if N <= 2:
                return [float('inf')] * N

            dist = [0.0] * N
            for m_idx in range(m):
                # Sort by objective m_idx
                sorted_idx = sorted(front_indices, key=lambda k: fitnesses[k][m_idx])
                dist[front_indices.index(sorted_idx[0])] = float('inf')
                dist[front_indices.index(sorted_idx[-1])] = float('inf')
                f_min = fitnesses[sorted_idx[0]][m_idx]
                f_max = fitnesses[sorted_idx[-1]][m_idx]
                if f_max == f_min:
                    continue
                for i in range(1, N - 1):
                    idx = front_indices.index(sorted_idx[i])
                    dist[idx] += (fitnesses[sorted_idx[i + 1]][m_idx] - fitnesses[sorted_idx[i - 1]][m_idx]) / (f_max - f_min)

            return dist

        # Differential Evolution parameters
        F_de = 0.8  # Mutation factor
        CR = 0.9    # Crossover rate

        while used < self.budget:
            batch = min(pop_size, self.budget - used)
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            for _ in range(batch):
                # Select three distinct random indices
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # DE Mutation
                v = X[r1] + F_de * (X[r2] - X[r3])
                
                # Binomial Crossover
                target_idx = rng.integers(pop_size)
                j_rand = rng.integers(n)
                crossover_mask = rng.random(n) < CR
                crossover_mask[j_rand] = True
                
                trial = np.where(crossover_mask, v, X[target_idx])
                trial = np.clip(trial, lo, hi)
                
                offspring_X.append(trial)
                offspring_F.append(problem(trial))
                used += 1
                
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort on combined
            combined_fronts = fast_non_dominated_sort(combined_F)
            
            # Select new population
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(front, combined_F)
                    # Sort front by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
