"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = 20
        used = 0
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used += pop_size
        
        # Non-dominated sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            nd = [0] * n
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            return fronts[:-1] # Remove last empty front

        # Crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return [float('inf')] * len(indices)
            
            m = self.n_obj
            n = len(indices)
            dist = [0.0] * n
            
            for k in range(m):
                sorted_idx = sorted(range(n), key=lambda i: F[indices[i]][k])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                
                f_range = F[indices[sorted_idx[-1]]][k] - F[indices[sorted_idx[0]]][k]
                if f_range == 0:
                    continue
                    
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (F[indices[sorted_idx[i+1]]][k] - F[indices[sorted_idx[i-1]]][k]) / f_range
            
            return dist

        # Selection
        def select_next_generation(F, pop_size):
            fronts = fast_non_dominated_sort(F)
            next_gen = []
            rank = [0] * len(F)
            
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
                if len(next_gen) + len(front) <= pop_size:
                    next_gen.extend(front)
                else:
                    # Need to select from this front based on crowding distance
                    needed = pop_size - len(next_gen)
                    cd = crowding_distance(F, front)
                    sorted_front = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in range(needed):
                        next_gen.append(front[sorted_front[i]])
                    break
            
            return next_gen

        # Crossover and Mutation
        def sbx_crossover(x1, x2):
            if rng.random() < 0.9:
                eta = 20.0
                c1 = np.zeros_like(x1)
                c2 = np.zeros_like(x2)
                for i in range(self.dim):
                    if rng.random() < 0.5:
                        c1[i] = x1[i]
                        c2[i] = x2[i]
                    else:
                        if x1[i] == x2[i]:
                            c1[i] = x1[i]
                            c2[i] = x2[i]
                        else:
                            beta1 = 2.0 * rng.random()
                            beta2 = 1.0 / beta1
                            alpha1 = 1.0 + 2.0 * (x1[i] - lo[i]) / (x2[i] - x1[i])
                            alpha2 = 1.0 + 2.0 * (x2[i] - hi[i]) / (x1[i] - x2[i])
                            
                            beta1q = min(1.0, (2.0 * beta1) ** (-1.0 / (eta + 1.0)))
                            beta2q = min(1.0, (2.0 * beta2) ** (-1.0 / (eta + 1.0)))
                            
                            c1[i] = 0.5 * (1.0 + beta1q) * x1[i] + 0.5 * (1.0 - beta1q) * x2[i]
                            c2[i] = 0.5 * (1.0 - beta2q) * x1[i] + 0.5 * (1.0 + beta2q) * x2[i]
            else:
                c1 = x1.copy()
                c2 = x2.copy()
            return c1, c2

        def mutate(x):
            eta = 20.0
            for i in range(self.dim):
                if rng.random() < 0.9:
                    y = x[i]
                    if y == lo[i] or y == hi[i]:
                        continue
                    delta1 = (y - lo[i]) / (hi[i] - lo[i])
                    delta2 = (hi[i] - y) / (hi[i] - lo[i])
                    rand = rng.random()
                    if rand < 0.5:
                        xy = 1.0 + (2.0 * (1.0 - delta1))
                        deltaq = (2.0 * rand + (1.0 - rand) * (xy - 1.0)) ** (1.0 / (eta + 1.0))
                    else:
                        xy = 1.0 + (2.0 * delta2)
                        deltaq = (2.0 * (1.0 - rand) + rand * (xy - 1.0)) ** (1.0 / (eta + 1.0))
                        deltaq = 1.0 - deltaq
                    x[i] = y + deltaq * (hi[i] - lo[i])
            return np.clip(x, lo, hi)

        # Main loop
        while used + pop_size <= self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring_X = []
            for i in range(0, pop_size, 2):
                # Tournament selection
                idx1 = rng.integers(pop_size)
                idx2 = rng.integers(pop_size)
                
                # Simple tournament: pick 2, evaluate rank (approximate by non-domination in current pop)
                # For efficiency, we just pick 2 random parents from current population
                p1 = X[idx1]
                p2 = X[idx2]
                
                c1, c2 = sbx_crossover(p1, p2)
                c1 = mutate(c1)
                c2 = mutate(c2)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
            
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size
            
            # Combine
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Select next generation
            next_indices = select_next_generation(combined_F, pop_size)
            X = combined_X[next_indices]
            F = combined_F[next_indices]

        return X
# EVOLVE-BLOCK-END
