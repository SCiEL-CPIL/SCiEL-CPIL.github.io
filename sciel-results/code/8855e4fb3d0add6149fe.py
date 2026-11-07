"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _fast_non_dominated_sort(self, F):
        """Return list of fronts (each a list of indices), sorted by dominance rank."""
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [list() for _ in range(n)]
        
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
        
        # Add any remaining points (shouldn't happen but safety)
        assigned = set()
        for front in fronts:
            assigned.update(front)
        if len(assigned) < n:
            remaining = [i for i in range(n) if i not in assigned]
            fronts.append(remaining)
        
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for points in a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        F_front = F[front_indices]
        m = F_front.shape[1]
        
        for k in range(m):
            min_val = F_front[:, k].min()
            max_val = F_front[:, k].max()
            if max_val - min_val < 1e-15:
                continue
            order = np.argsort(F_front[:, k])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for i in range(1, n - 1):
                dist[order[i]] += (F_front[order[i + 1], k] - F_front[order[i - 1], k]) / (max_val - min_val)
        
        return dist

    def _tournament_select(self, rank, cd, F, X, k=2):
        """Tournament selection based on rank and crowding distance."""
        n = len(F)
        idx1 = np.random.randint(n)
        idx2 = np.random.randint(n)
        if rank[idx1] < rank[idx2]:
            return idx1
        elif rank[idx2] < rank[idx1]:
            return idx2
        else:
            if cd[idx1] > cd[idx2]:
                return idx1
            else:
                return idx2

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        pop_size = min(50, self.budget // 4)
        pop_size = max(pop_size, 10)
        
        # Initialize population
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)
            
            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Compute crowding distance for each front
            cd = np.zeros(len(F))
            for front in fronts:
                front_cd = self._crowding_distance(F, front)
                for j, idx in enumerate(front):
                    cd[idx] = front_cd[j]
            
            # Generate offspring
            n_offspring = pop_size
            offspring_X = []
            offspring_F = []
            
            for _ in range(n_offspring):
                if used >= self.budget:
                    break
                
                # Select parents
                p1 = self._tournament_select(rank, cd, F, X)
                p2 = self._tournament_select(rank, cd, F, X)
                
                # Simulated binary crossover (SBX)
                if rng.random() < 0.9:
                    beta = 15
                    child1 = np.zeros(self.dim)
                    child2 = np.zeros(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            if abs(X[p1, d] - X[p2, d]) < 1e-15:
                                child1[d] = X[p1, d]
                                child2[d] = X[p2, d]
                            else:
                                u = rng.random()
                                if u <= 0.5:
                                    alpha = 2 * u
                                    factor1 = (alpha + 1) ** (-1.0 / (beta + 1))
                                    factor2 = ((1 - alpha) + 1) ** (-1.0 / (beta + 1))
                                else:
                                    alpha = 2 * (1 - u)
                                    factor1 = (alpha + 1) ** (-1.0 / (beta + 1))
                                    factor2 = ((1 - alpha) + 1) ** (-1.0 / (beta + 1))
                                
                                x1 = min(X[p1, d], X[p2, d])
                                x2 = max(X[p1, d], X[p2, d])
                                child1[d] = 0.5 * ((1 + factor1) * x1 + (1 - factor1) * x2)
                                child2[d] = 0.5 * ((1 + factor2) * x2 + (1 - factor2) * x1)
                        else:
                            child1[d] = X[p1, d]
                            child2[d] = X[p2, d]
                    
                    # Polynomial mutation
                    eta = 20
                    for d in range(self.dim):
                        if rng.random() < 0.9:
                            u = rng.random()
                            if u < 0.5:
                                delta_q = (2 * u) ** (1.0 / (eta + 1)) - 1
                            else:
                                delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                            child1[d] += delta_q * (hi[d] - lo[d])
                        
                        if rng.random() < 0.9:
                            u = rng.random()
                            if u < 0.5:
                                delta_q = (2 * u) ** (1.0 / (eta + 1)) - 1
                            else:
                                delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                            child2[d] += delta_q * (hi[d] - lo[d])
                    
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    
                    fc1 = problem(child1)
                    used += 1
                    offspring_X.append(child1)
                    offspring_F.append(fc1)
                    
                    if used < self.budget:
                        fc2 = problem(child2)
                        used += 1
                        offspring_X.append(child2)
                        offspring_F.append(fc2)
                else:
                    # Just mutate
                    child = X[p1].copy()
                    for d in range(self.dim):
                        if rng.random() < 0.9:
                            u = rng.random()
                            if u < 0.5:
                                delta_q = (2 * u) ** (1.0 / (21)) - 1
                            else:
                                delta_q = 1 - (2 * (1 - u)) ** (1.0 / (21))
                            child[d] += delta_q * (hi[d] - lo[d])
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    used += 1
                    offspring_X.append(child)
                    offspring_F.append(fc)
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_F)
            
            # Non-dominated sort on combined
            fronts = self._fast_non_dominated_sort(combined_F)
            
            # Assign ranks
            rank = np.zeros(n_combined, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Compute crowding distance
            cd = np.zeros(n_combined)
            for front in fronts:
                front_cd = self._crowding_distance(combined_F, front)
                for j, idx in enumerate(front):
                    cd[idx] = front_cd[j]
            
            # Select next generation
            new_X = np.zeros((pop_size, self.dim))
            new_F = np.zeros((pop_size, self.n_obj))
            count = 0
            for front in fronts:
                if count + len(front) <= pop_size:
                    for idx in front:
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                else:
                    # Sort front by crowding distance (descending)
                    front_cd = cd[front]
                    order = np.argsort(-front_cd)
                    remaining = pop_size - count
                    for k in range(min(remaining, len(front))):
                        idx = front[order[k]]
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                    break
            
            X = new_X[:count]
            F = new_F[:count]
            
            # Update archive of all evaluated points
            all_X.append(X.copy())
            all_F.append(F.copy())
        
        return X
# EVOLVE-BLOCK-END
