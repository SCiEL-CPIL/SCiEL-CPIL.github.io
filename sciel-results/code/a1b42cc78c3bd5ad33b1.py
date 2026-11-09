"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II-like strategy with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Return list of fronts (list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        fronts = []
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                # p dominates q?
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    nd[p] += 1
                    # Actually, nd[p] counts number of solutions that dominate p
                    # So if q dominates p, nd[p] should be incremented.
                    # My condition above: q dominates p.
        
        # Correction: The loop above is O(n^2) and a bit messy. Let's do it properly.
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                # Check if p dominates q
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                # Check if q dominates p
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    nd[p] += 1
        
        # First front
        front = [i for i in range(n) if nd[i] == 0]
        if not front:
            # All dominated? Should not happen for first front unless n=0
            front = [0]
        
        fronts.append(front)
        i = 0
        while front:
            next_front = []
            for p in front:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            front = next_front
            i += 1
        return fronts

    def _crowding_distance(self, F, indices):
        """Calculate crowding distance for a front."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        F_sub = F[indices]
        
        for m in range(self.n_obj):
            # Sort by objective m
            order = np.argsort(F_sub[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            
            min_val = F_sub[order[0], m]
            max_val = F_sub[order[-1], m]
            if max_val == min_val:
                continue
                
            for k in range(1, n - 1):
                dist[order[k]] += (F_sub[order[k + 1], m] - F_sub[order[k - 1], m]) / (max_val - min_val)
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Initial population
        pop_size = min(50, self.budget // 2)
        X = lo + span * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Reference point for hypervolume (not strictly needed for NSGA-II, but good for tracking)
        ref_point = np.array([1.1, 1.3, 1.1])
        
        while used < self.budget:
            # Check deadline
            if time.time() > deadline - 10.0:
                break
            
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)
            
            # Selection: fill next generation
            next_indices = []
            for front in fronts:
                if len(next_indices) + len(front) <= pop_size:
                    next_indices.extend(front)
                else:
                    # Truncate using crowding distance
                    need = pop_size - len(next_indices)
                    dist = self._crowding_distance(F, front)
                    order = np.argsort(dist)[::-1]
                    next_indices.extend([front[i] for i in order[:need]])
                    break
            
            # Crossover and Mutation
            # Simple SBX-like or BLX-alpha, let's use a simple uniform crossover + gaussian mutation
            offspring = []
            while len(offspring) < pop_size:
                # Tournament selection
                p1 = self._tournament_select(F, fronts)
                p2 = self._tournament_select(F, fronts)
                
                # Crossover
                if np.random.random() < 0.9:
                    mask = np.random.random(self.dim) < 0.5
                    c1 = np.where(mask, X[p1], X[p2])
                    c2 = np.where(mask, X[p2], X[p1])
                else:
                    c1 = X[p1].copy()
                    c2 = X[p2].copy()
                
                # Mutation
                for c in [c1, c2]:
                    if np.random.random() < 0.9:
                        # Gaussian mutation
                        c += 0.2 * span * np.random.standard_normal(self.dim)
                        c = np.clip(c, lo, hi)
                
                offspring.append(c1)
                offspring.append(c2)
                
                if len(offspring) >= pop_size:
                    break
            
            offspring = np.array(offspring[:pop_size])
            
            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring])
            used += pop_size
            
            # Combine
            X = np.vstack([X, offspring])
            F = np.vstack([F, F_off])
            
            # Selection for next generation
            fronts = self._fast_non_dominated_sort(F)
            next_gen_indices = []
            for front in fronts:
                if len(next_gen_indices) + len(front) <= pop_size:
                    next_gen_indices.extend(front)
                else:
                    need = pop_size - len(next_gen_indices)
                    dist = self._crowding_distance(F, front)
                    order = np.argsort(dist)[::-1]
                    next_gen_indices.extend([front[i] for i in order[:need]])
                    break
            
            X = X[next_gen_indices]
            F = F[next_gen_indices]
        
        return X

    def _tournament_select(self, F, fronts):
        """Tournament selection based on front rank."""
        rank = np.zeros(len(F), dtype=int)
        for i, front in enumerate(fronts):
            for idx in front:
                rank[idx] = i
        
        t1 = np.random.randint(len(F))
        t2 = np.random.randint(len(F))
        if rank[t1] < rank[t2]:
            return t1
        elif rank[t2] < rank[t1]:
            return t2
        else:
            # Same front, pick randomly
            return np.random.choice([t1, t2])
# EVOLVE-BLOCK-END
