"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II multi-objective optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n_obj = self.n_obj
        dim = self.dim

        # Population size
        N = 15

        # Initial population
        X = lo + span * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        # Helper: Non-dominated sorting
        def fast_non_dominated_sorting(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = [0] * n
            rank = [0] * n
            fronts = []
            current_front = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominates[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominates[j] += 1
                
                if n_dominates[i] == 0:
                    rank[i] = 0
                    current_front.append(i)

            if current_front:
                fronts.append(current_front)

            k = 1
            while fronts and len(fronts[-1]) > 0:
                next_front = []
                for i in fronts[-1]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            rank[j] = k
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                else:
                    break
                k += 1
            
            return rank, fronts

        # Helper: Crowding distance
        def crowding_distance_assignment(F, indices):
            n = len(indices)
            if n == 0:
                return {}
            
            cd = {i: 0.0 for i in indices}
            if n == 1:
                cd[indices[0]] = float('inf')
                return cd

            for m in range(n_obj):
                sorted_indices = sorted(indices, key=lambda i: F[i][m])
                cd[sorted_indices[0]] = float('inf')
                cd[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0]][m]
                f_max = F[sorted_indices[-1]][m]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                    
                for i in range(1, n - 1):
                    cd[sorted_indices[i]] += (F[sorted_indices[i+1]][m] - F[sorted_indices[i-1]][m]) / f_range
            
            return cd

        # Helper: Binary tournament selection
        def binary_tournament(pop_X, pop_F, ranks, cds):
            i1 = rng.integers(0, len(pop_X))
            i2 = rng.integers(0, len(pop_X))
            
            if ranks[i1] < ranks[i2]:
                return pop_X[i1].copy()
            elif ranks[i2] < ranks[i1]:
                return pop_X[i2].copy()
            else:
                if cds[i1] > cds[i2]:
                    return pop_X[i1].copy()
                else:
                    return pop_X[i2].copy()

        # SBX crossover
        def sbx_crossover(x1, x2, eta=20.0):
            if rng.random() < 0.5:
                return x1.copy(), x2.copy()
            child1 = np.zeros(dim)
            child2 = np.zeros(dim)
            for i in range(dim):
                if abs(x1[i] - x2[i]) < 1e-14:
                    child1[i] = x2[i]
                    child2[i] = x1[i]
                else:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return child1, child2

        # Polynomial mutation
        def polynomial_mutation(x, eta=20.0):
            x = x.copy()
            for i in range(dim):
                if rng.random() < 1.0 / dim:
                    u = rng.random()
                    lo_i = lo[i]
                    hi_i = hi[i]
                    if u < 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta_q = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    x[i] += delta_q * (hi_i - lo_i)
                    x[i] = np.clip(x[i], lo_i, hi_i)
            return x

        # Initial sorting
        rank, _ = fast_non_dominated_sorting(F)
        cd = crowding_distance_assignment(F, list(range(N)))

        # Main loop
        while used < self.budget:
            # Generate offspring
            offspring_X = []
            while len(offspring_X) < N and used + (N - len(offspring_X)) <= self.budget:
                 parent1 = binary_tournament(X, F, rank, cd)
                 parent2 = binary_tournament(X, F, rank, cd)
                 c1, c2 = sbx_crossover(parent1, parent2)
                 c1 = polynomial_mutation(c1)
                 c2 = polynomial_mutation(c2)
                 c1 = np.clip(c1, lo, hi)
                 c2 = np.clip(c2, lo, hi)
                 offspring_X.append(c1)
                 if len(offspring_X) < N:
                     offspring_X.append(c2)

            if not offspring_X:
                break

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += len(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sorting
            rank, fronts = fast_non_dominated_sorting(combined_F)
            
            # Select next generation
            new_indices = []
            for k in range(len(fronts)):
                if len(new_indices) + len(fronts[k]) <= N:
                    new_indices.extend(fronts[k])
                else:
                    # Crowding distance for this front
                    front_indices = fronts[k]
                    cd_front = crowding_distance_assignment(combined_F, front_indices)
                    front_indices_sorted = sorted(front_indices, key=lambda i: cd_front[i], reverse=True)
                    need = N - len(new_indices)
                    new_indices.extend(front_indices_sorted[:need])
                    break
            
            X = combined_X[new_indices]
            F = combined_F[new_indices]
            
            # Recalculate rank and cd for the new population for next iteration
            rank, _ = fast_non_dominated_sorting(F)
            cd = crowding_distance_assignment(F, list(range(N)))

        return X
# EVOLVE-BLOCK-END
