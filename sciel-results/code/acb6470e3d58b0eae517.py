"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """Return (fronts, dominance_counts) where fronts is list of lists of indices."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int)
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        current_front = [i for i in range(n) if n_dom[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for given indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        F_sub = F[indices]
        m = self.n_obj
        
        for k in range(m):
            order = np.argsort(F_sub[:, k])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            f_range = F_sub[order[-1], k] - F_sub[order[0], k]
            if f_range > 1e-12:
                for i in range(1, n - 1):
                    dist[order[i]] += (F_sub[order[i + 1], k] - F_sub[order[i - 1], k]) / f_range
        
        return dist

    def _selection_tournament(self, F, fronts, n_select):
        """Tournament selection based on non-dominated rank and crowding distance."""
        n = len(F)
        # Assign ranks
        rank = np.zeros(n, dtype=int)
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r + 1
        
        # Compute crowding distance for each front
        cd = np.zeros(n)
        for front in fronts:
            if len(front) > 0:
                cd[front] = self._crowding_distance(F, front)
        
        selected = []
        for _ in range(n_select):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if rank[i] < rank[j]:
                selected.append(i)
            elif rank[i] > rank[j]:
                selected.append(j)
            else:
                if cd[i] > cd[j]:
                    selected.append(i)
                else:
                    selected.append(j)
        
        return selected

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0  # 10s margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        # Population size: balance exploration and computation
        pop_size = min(100, max(20, self.budget // 2))
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Main NSGA-II loop
        while used < self.budget:
            # Check time
            if time.time() - start_time > deadline - 5:
                break
            
            # Non-dominated sort
            fronts = self._non_dominated_sort(F)
            
            # Create offspring
            n_offspring = pop_size
            parents = self._selection_tournament(F, fronts, 2 * n_offspring)
            
            offspring_X = np.empty((n_offspring, self.dim))
            offspring_F = np.empty((n_offspring, self.n_obj))
            
            for i in range(0, n_offspring, 1):
                p1 = parents[2 * i]
                p2 = parents[2 * i + 1]
                
                # SBX crossover
                if rng.random() < 0.9:
                    eta = 20.0
                    u = rng.random(self.dim)
                    v = rng.random(self.dim)
                    
                    c1 = np.zeros(self.dim)
                    c2 = np.zeros(self.dim)
                    for d in range(self.dim):
                        if u[d] <= 0.5:
                            beta = (2 * u[d]) ** (1 / (eta + 1))
                        else:
                            beta = (2 * (1 - u[d])) ** (1 / (eta + 1))
                        c1[d] = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])
                        
                        if v[d] <= 0.5:
                            beta = (2 * v[d]) ** (1 / (eta + 1))
                        else:
                            beta = (2 * (1 - v[d])) ** (1 / (eta + 1))
                        c2[d] = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])
                    
                    # Polynomial mutation
                    for d in range(self.dim):
                        if rng.random() < 0.9:
                            u = rng.random()
                            if u < 0.5:
                                delta = (2 * u) ** (1 / (eta + 1)) - 1
                            else:
                                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                            c1[d] += delta * (hi[d] - lo[d])
                        if rng.random() < 0.9:
                            u = rng.random()
                            if u < 0.5:
                                delta = (2 * u) ** (1 / (eta + 1)) - 1
                            else:
                                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                            c2[d] += delta * (hi[d] - lo[d])
                    
                    offspring_X[i, :] = np.clip(c1, lo, hi)
                    if i + 1 < n_offspring:
                        offspring_X[i + 1, :] = np.clip(c2, lo, hi)
                else:
                    offspring_X[i, :] = X[p1]
                    if i + 1 < n_offspring:
                        offspring_X[i + 1, :] = X[p2]
                
                # Evaluate
                offspring_F[i] = problem(offspring_X[i])
                used += 1
                if i + 1 < n_offspring and used < self.budget:
                    offspring_F[i + 1] = problem(offspring_X[i + 1])
                    used += 1
                    i += 1  # Skip next iteration
                if used >= self.budget:
                    break
            
            # Combine and select next generation
            X_all = np.vstack([X, offspring_X[:used - pop_size]])
            F_all = np.vstack([F, offspring_F[:used - pop_size]])
            
            fronts = self._non_dominated_sort(F_all)
            
            new_X = np.empty((pop_size, self.dim))
            new_F = np.empty((pop_size, self.n_obj))
            count = 0
            
            for front in fronts:
                if count + len(front) <= pop_size:
                    for idx in front:
                        new_X[count] = X_all[idx]
                        new_F[count] = F_all[idx]
                        count += 1
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - count
                    cd = self._crowding_distance(F_all, front)
                    sorted_front = [front[i] for i in np.argsort(-cd)]
                    for idx in sorted_front[:remaining]:
                        new_X[count] = X_all[idx]
                        new_F[count] = F_all[idx]
                        count += 1
                    break
            
            X = new_X[:count]
            F = new_F[:count]
            if count < pop_size:
                # Fill remaining with random
                for i in range(count, pop_size):
                    X[i] = lo + (hi - lo) * rng.random(self.dim)
                    F[i] = problem(X[i])
                    used += 1
        
        return X
# EVOLVE-BLOCK-END
