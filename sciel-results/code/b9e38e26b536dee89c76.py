"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II-inspired approach with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _non_dominated_sort(self, F):
        """Return list of fronts, each a list of indices."""
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
        
        fronts = []
        current_front = [i for i in range(n) if domination_count[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
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
        
        for m in range(self.n_obj):
            order = np.argsort(F_sub[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            f_range = F_sub[order[-1], m] - F_sub[order[0], m]
            if f_range > 0:
                for k in range(1, n - 1):
                    dist[order[k]] += (F_sub[order[k + 1], m] - F_sub[order[k - 1], m]) / f_range
        
        return dist

    def _select_parents(self, F, X, n_parents):
        """Select parents using tournament selection on non-dominated fronts."""
        fronts = self._non_dominated_sort(F)
        
        # Build a rank array
        rank = np.zeros(len(F), dtype=int)
        for i, front in enumerate(fronts):
            for idx in front:
                rank[idx] = i
        
        # Tournament selection
        parents = []
        while len(parents) < n_parents:
            # Pick two random candidates
            i, j = np.random.choice(len(F), 2, replace=False)
            if rank[i] < rank[j]:
                parents.append(i)
            elif rank[j] < rank[i]:
                parents.append(j)
            else:
                # Same rank, use crowding distance
                front_idx = rank[i]
                front = fronts[front_idx]
                cd = self._crowding_distance(F, front)
                pos_i = np.where(np.array(front) == i)[0][0]
                pos_j = np.where(np.array(front) == j)[0][0]
                if cd[pos_i] > cd[pos_j]:
                    parents.append(i)
                else:
                    parents.append(j)
        
        return parents

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        # Initial population
        pop_size = min(50, self.budget // 4)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Archive of all non-dominated points found
        archive_X = [X[0].copy()]
        archive_F = [F[0].copy()]
        
        while used < self.budget:
            # Check time
            if time.time() > deadline - 5:
                break
            
            # Generate offspring
            n_offspring = min(pop_size, self.budget - used)
            
            # Select parents
            parent_indices = self._select_parents(F, X, n_offspring * 2)
            
            offspring_X = []
            offspring_F = []
            
            for k in range(n_offspring):
                p1_idx = parent_indices[k * 2]
                p2_idx = parent_indices[k * 2 + 1]
                
                # SBX-like crossover
                if rng.random() < 0.9:
                    beta = 20  # Distribution index
                    u = rng.random()
                    if u <= 0.5:
                        alpha = (u * 2) ** (1 / (beta + 1))
                    else:
                        alpha = (1 - u * 2) ** (1 / (beta + 1))
                    
                    child1 = 0.5 * ((1 + alpha) * X[p1_idx] + (1 - alpha) * X[p2_idx])
                    child2 = 0.5 * ((1 - alpha) * X[p1_idx] + (1 + alpha) * X[p2_idx])
                else:
                    child1 = X[p1_idx].copy()
                    child2 = X[p2_idx].copy()
                
                # Mutation with adaptive rate
                mut_rate = 1.0 / self.dim
                for child in [child1, child2]:
                    mask = rng.random(self.dim) < mut_rate
                    if np.any(mask):
                        # Polynomial mutation
                        eta = 20
                        u = rng.random(np.sum(mask))
                        for idx, m_idx in enumerate(np.where(mask)[0]):
                            if u[idx] <= 0.5:
                                delta = (2 * u[idx]) ** (1 / (eta + 1)) - 1
                            else:
                                delta = 1 - (2 * (1 - u[idx])) ** (1 / (eta + 1))
                            child[m_idx] += delta * (hi[m_idx] - lo[m_idx])
                
                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)
                
                offspring_X.append(child1)
                offspring_X.append(child2)
            
            # Evaluate offspring (batch)
            for ox in offspring_X:
                if used >= self.budget:
                    break
                of = problem(ox)
                offspring_F.append(of)
                used += 1
                archive_X.append(ox.copy())
                archive_F.append(of.copy())
            
            if not offspring_F:
                break
            
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X[:len(offspring_F)])])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            fronts = self._non_dominated_sort(combined_F)
            
            # Select next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front using crowding distance
                    needed = pop_size - len(new_X)
                    cd = self._crowding_distance(combined_F, front)
                    sorted_idx = np.argsort(-cd)
                    for i in range(needed):
                        idx = front[sorted_idx[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
        
        # Return archive (or last population)
        return X
# EVOLVE-BLOCK-END
