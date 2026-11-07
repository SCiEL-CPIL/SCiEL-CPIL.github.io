"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II-style approach for ZDT3."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Initial population
        pop_size = min(100, self.budget // 2)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II parameters
        mutation_rate = 0.3
        crossover_rate = 0.9
        sigma = 0.1
        
        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            fronts = [[]]
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_count[p] += 1
                if n_count[p] == 0:
                    fronts[0].append(p)
            
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            next_front.append(q)
                i += 1
                if next_front:
                    fronts.append(next_front)
            
            return fronts
        
        def crowding_distance(F, front_indices):
            """Calculate crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            
            dist = np.zeros(n)
            for m in range(n_obj):
                obj_vals = F[front_indices, m]
                min_val, max_val = obj_vals.min(), obj_vals.max()
                if max_val - min_val < 1e-10:
                    continue
                sorted_idx = np.argsort(obj_vals)
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / (max_val - min_val)
            
            return dist
        
        def tournament_select(F, fronts, k=2):
            """Tournament selection based on rank and crowding distance."""
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
            
            # Calculate crowding distances per front
            cd = np.zeros(n)
            for front in fronts:
                cd[front] = crowding_distance(F, front)
            
            # Tournament
            candidates = rng.integers(n, size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if ranks[c] < ranks[best]:
                    best = c
                elif ranks[c] == ranks[best] and cd[c] > cd[best]:
                    best = c
            return best
        
        def crossover(x1, x2):
            """SBX crossover."""
            if rng.random() > crossover_rate:
                return x1.copy(), x2.copy()
            
            child1, child2 = x1.copy(), x2.copy()
            for i in range(dim):
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (2 + 1))
                else:
                    beta = (2 * (1 - u)) ** (1 / (2 + 1))
                
                if x1[i] == x2[i]:
                    child1[i] = x2[i]
                    child2[i] = x1[i]
                else:
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            
            return child1, child2
        
        def mutate(x):
            """Polynomial mutation."""
            x_new = x.copy()
            for i in range(dim):
                if rng.random() < mutation_rate:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / 21) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / 21)
                    x_new[i] = x[i] + delta * (hi[i] - lo[i])
                    x_new[i] = np.clip(x_new[i], lo[i], hi[i])
            return x_new
        
        # Main evolution loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Create offspring population
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size and used + len(offspring_X) < self.budget:
                # Check deadline
                if time.time() > deadline:
                    break
                
                # Selection
                p1_idx = tournament_select(F, fronts)
                p2_idx = tournament_select(F, fronts)
                
                # Crossover
                c1, c2 = crossover(X[p1_idx], X[p2_idx])
                
                # Mutation
                c1 = mutate(c1)
                c2 = mutate(c2)
                
                # Evaluate
                f1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                if used + len(offspring_X) < self.budget and len(offspring_X) < pop_size:
                    f2 = problem(c2)
                    used += 1
                    offspring_X.append(c2)
                    offspring_F.append(f2)
            
            if not offspring_X:
                break
            
            offspring_F = np.array(offspring_F)
            
            # Combine parents and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])
            combined_n = len(combined_X)
            
            # Fast non-dominated sorting
            fronts = fast_non_dominated_sort(combined_F)
            
            # Select next generation
            next_X = []
            next_F = []
            remaining = pop_size
            
            for front in fronts:
                if remaining <= 0:
                    break
                if len(front) <= remaining:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    remaining -= len(front)
                else:
                    # Need to truncate using crowding distance
                    cd = crowding_distance(combined_F, front)
                    sorted_front = np.argsort(-cd)
                    for i in range(remaining):
                        idx = front[sorted_front[i]]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    remaining = 0
            
            if not next_X:
                break
            
            X = np.array(next_X)
            F = np.array(next_F)
            
            # Recalculate fronts for next iteration
            fronts = fast_non_dominated_sort(F)
        
        return X
# EVOLVE-BLOCK-END
