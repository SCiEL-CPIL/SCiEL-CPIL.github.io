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
        if n == 0:
            return []
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

    def _tournament_select(self, rank, cd, rng, k=2):
        """Tournament selection based on rank and crowding distance."""
        n = len(rank)
        idx1 = rng.integers(n)
        idx2 = rng.integers(n)
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
        
        pop_size = min(40, self.budget // 5)
        pop_size = max(pop_size, 10)
        
        # Initialize population
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
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
                p1 = self._tournament_select(rank, cd, rng)
                p2 = self._tournament_select(rank, cd, rng)
                
                # Simulated binary crossover (SBX)
                if rng.random() < 0.9:
                    beta = 15
                    child = np.zeros(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            if abs(X[p1, d] - X[p2, d]) < 1e-15:
                                child[d] = X[p1, d]
                            else:
                                u = rng.random()
                                if u <= 0.5:
                                    alpha = 2 * u
                                    factor = (alpha + 1) ** (-1.0 / (beta + 1))
                                else:
                                    alpha = 2 * (1 - u)
                                    factor = (alpha + 1) ** (-1.0 / (beta + 1))
                                
                                x1 = min(X[p1, d], X[p2, d])
                                x2 = max(X[p1, d], X[p2, d])
                                child[d] = 0.5 * ((1 + factor) * x1 + (1 - factor) * x2)
                        else:
                            if abs(X[p1, d] - X[p2, d]) < 1e-15:
                                child[d] = X[p2, d]
                            else:
                                u = rng.random()
                                if u <= 0.5:
                                    alpha = 2 * u
                                    factor = (alpha + 1) ** (-1.0 / (beta + 1))
                                else:
                                    alpha = 2 * (1 - u)
                                    factor = (alpha + 1) ** (-1.0 / (beta + 1))
                                
                                x1 = min(X[p1, d], X[p2, d])
                                x2 = max(X[p1, d], X[p2, d])
                                child[d] = 0.5 * ((1 + factor) * x2 + (1 - factor) * x1)
                else:
                    child = X[p1].copy()
                
                # Polynomial mutation
                if rng.random() < 0.9:
                    eta = 20
                    for d in range(self.dim):
                        u = rng.random()
                        if u < 0.5:
                            delta1 = (2 * u) ** (1.0 / (eta + 1)) - 1
                        else:
                            delta1 = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                        
                        if abs(hi[d] - lo[d]) < 1e-15:
                            continue
                        child[d] = child[d] + delta1 * (hi[d] - lo[d])
                        child[d] = np.clip(child[d], lo[d], hi[d])
                
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
            
            # Non-dominated sorting on combined
            combined_fronts = self._fast_non_dominated_sort(combined_F)
            
            # Assign ranks
            combined_rank = np.zeros(len(combined_F), dtype=int)
            for i, front in enumerate(combined_fronts):
                for idx in front:
                    combined_rank[idx] = i
            
            # Compute crowding distance for each front
            combined_cd = np.zeros(len(combined_F))
            for front in combined_fronts:
                front_cd = self._crowding_distance(combined_F, front)
                for j, idx in enumerate(front):
                    combined_cd[idx] = front_cd[j]
            
            # Select new population
            new_X = np.zeros((pop_size, self.dim))
            new_F = np.zeros((pop_size, self.n_obj))
            new_indices = []
            
            for front in combined_fronts:
                if len(new_indices) + len(front) <= pop_size:
                    new_indices.extend(front)
                else:
                    # Sort by crowding distance and take remaining
                    front_cd_vals = [combined_cd[i] for i in front]
                    front_sorted = [front[i] for i in np.argsort(front_cd_vals)[::-1]]
                    remaining = pop_size - len(new_indices)
                    new_indices.extend(front_sorted[:remaining])
                    break
            
            new_X = combined_X[new_indices]
            new_F = combined_F[new_indices]
            
            X = new_X
            F = new_F
        
        return X
# EVOLVE-BLOCK-END
