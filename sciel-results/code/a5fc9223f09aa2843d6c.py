"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Population size: balance between exploration and evaluation budget
        pop_size = min(50, max(20, self.budget // 4))

        # Initial population: uniform random
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Returns list of fronts, each containing indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = [[]]
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                        n_dom[p] += 1
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dom[q] += 1
                
                if n_dom[p] == 0:
                    fronts[0].append(p)
            
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            
            # Remove last empty front
            if not fronts[-1]:
                fronts.pop()
            
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for indices in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            dist = np.zeros(n)
            m = self.n_obj
            
            for k in range(m):
                sorted_idx = np.argsort(F[front_indices, k])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                
                f_min = F[front_indices[sorted_idx[0]], k]
                f_max = F[front_indices[sorted_idx[-1]], k]
                f_range = f_max - f_min
                
                if f_range > 1e-12:
                    for i in range(1, n - 1):
                        dist[sorted_idx[i]] += (F[front_indices[sorted_idx[i+1]], k] - F[front_indices[sorted_idx[i-1]], k]) / f_range
            
            return dist

        def tournament_select(fronts, F, n_select):
            """Select parents using tournament selection based on fronts and crowding."""
            # Compute crowding for each front
            front_crowd = {}
            for front in fronts:
                if front:
                    front_crowd[tuple(front)] = crowding_distance(F, front)
            
            parents = []
            while len(parents) < n_select:
                # Tournament of size 2
                i, j = rng.integers(0, len(F)), rng.integers(0, len(F))
                
                # Find front ranks
                rank_i, rank_j = -1, -1
                for r, front in enumerate(fronts):
                    if i in front:
                        rank_i = r
                        break
                    if j in front:
                        rank_j = r
                        break
                
                if rank_i < rank_j:
                    parent = i
                elif rank_j < rank_i:
                    parent = j
                else:
                    # Same front, use crowding distance
                    front = fronts[rank_i]
                    crowd = front_crowd[tuple(front)]
                    ci = crowd[front.index(i)]
                    cj = crowd[front.index(j)]
                    parent = i if ci > cj else j
                
                parents.append(parent)
            
            return parents

        def crossover(parent1, parent2):
            """SBX crossover."""
            if rng.random() < 0.9:
                # Simulated binary crossover
                child1 = np.zeros(self.dim)
                child2 = np.zeros(self.dim)
                for i in range(self.dim):
                    if rng.random() < 0.5:
                        # No crossover for this gene
                        child1[i] = parent1[i]
                        child2[i] = parent2[i]
                    else:
                        # SBX
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (2 + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (2 + 1))
                        
                        child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                        child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                        
                        # Clip to bounds
                        child1[i] = np.clip(child1[i], lo[i], hi[i])
                        child2[i] = np.clip(child2[i], lo[i], hi[i])
                return child1, child2
            else:
                return parent1.copy(), parent2.copy()

        def mutate(x, rate=0.1):
            """Polynomial mutation with adaptive rate."""
            child = x.copy()
            for i in range(self.dim):
                if rng.random() < rate:
                    u = rng.random()
                    if u <= 0.5:
                        delta = (2 * u) ** (1 / (1 + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (1 + 1))
                    
                    child[i] = x[i] + delta * (hi[i] - lo[i])
                    child[i] = np.clip(child[i], lo[i], hi[i])
            return child

        # Main loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Evaluate fronts
            fronts = fast_non_dominated_sort(F)
            
            # Select parents and create offspring
            offspring_X = []
            offspring_F = []
            
            for _ in range(pop_size // 2):
                if time.time() > deadline:
                    break
                
                p1_idx, p2_idx = tournament_select(fronts, F, 2)
                p1, p2 = X[p1_idx], X[p2_idx]
                
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1)
                c2 = mutate(c2)
                
                fc1 = problem(c1)
                used += 1
                fc2 = problem(c2)
                used += 1
                
                offspring_X.append(c1)
                offspring_F.append(fc1)
                offspring_X.append(c2)
                offspring_F.append(fc2)
            
            if not offspring_X:
                break
            
            # Combine parent and offspring
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.vstack([F, np.array(offspring_F)])
            
            # Select best pop_size from combined
            fronts_combined = fast_non_dominated_sort(F_combined)
            
            new_X = []
            new_F = []
            for front in fronts_combined:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                else:
                    # Need to select from this front using crowding
                    remaining = pop_size - len(new_X)
                    crowd = crowding_distance(F_combined, front)
                    sorted_front = sorted(front, key=lambda i: crowd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
