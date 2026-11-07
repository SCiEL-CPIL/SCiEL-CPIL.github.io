import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        # Smaller population for more generations
        self.pop_size = 12
        self.deadline = time.time() + 280  # 300s limit, 20s safety margin

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        n_obj = self.n_obj
        
        rng = np.random.default_rng(42)
        
        # Initialization
        X = lo + (hi - lo) * rng.random((self.pop_size, d))
        F = np.array([problem(x) for x in X])
        used = self.pop_size
        
        # Track all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        while used + self.pop_size <= self.budget:
            if time.time() > self.deadline:
                break
                
            # Calculate ranks and CD for current population X
            ranks, cd = self._calculate_rank_cd(F)
            
            # Create offspring
            Q = np.zeros((self.pop_size, d))
            FQ = np.zeros((self.pop_size, n_obj))
            
            for i in range(self.pop_size):
                p1 = self._tournament(ranks, cd, rng)
                p2 = self._tournament(ranks, cd, rng)
                
                x1 = X[p1]
                x2 = X[p2]
                
                # Crossover: BLX-Alpha
                alpha = 0.5
                low = np.minimum(x1, x2) - alpha * np.abs(x1 - x2)
                high = np.maximum(x1, x2) + alpha * np.abs(x1 - x2)
                child = rng.uniform(low, high, size=d)
                child = np.clip(child, lo, hi)
                
                # Mutation
                for k in range(d):
                    if rng.random() < 1.0/d:
                        child[k] += 0.1 * (hi[k] - lo[k]) * rng.standard_normal()
                        child[k] = np.clip(child[k], lo, hi)
                
                Q[i] = child
                FQ[i] = problem(child)
                used += 1
                
                if time.time() > self.deadline:
                    break
            
            if used >= self.budget:
                break
            
            # Combine and select
            R_X = np.vstack([X, Q])
            R_F = np.vstack([F, FQ])
            
            X, F = self._nsga2_select(R_X, R_F)
            
            # Track all points
            all_X.append(Q[:used - (self.pop_size - (self.pop_size - 1))].copy() if used > self.pop_size else np.empty((0, d)))
            all_F.append(FQ[:used - (self.pop_size - (self.pop_size - 1))].copy() if used > self.pop_size else np.empty((0, n_obj)))
        
        # Tail: use remaining budget for local refinement
        if used < self.budget:
            # Get current non-dominated front
            front_idx = self._get_non_dominated_indices(F)
            front_X = X[front_idx]
            front_F = F[front_idx]
            
            # Do local Nelder-Mead refinement around front points
            remaining = self.budget - used
            if len(front_X) > 0 and remaining > 0:
                # Sample from front points
                n_local = min(remaining, len(front_X) * 5)
                local_budget = n_local
                
                # Create local search points
                local_X = np.zeros((n_local, d))
                local_F = np.zeros((n_local, n_obj))
                
                for i in range(n_local):
                    # Pick a random front point
                    fp_idx = rng.integers(len(front_X))
                    center = front_X[fp_idx].copy()
                    
                    # Small perturbation
                    for k in range(d):
                        if rng.random() < 0.8:
                            center[k] += 0.02 * (hi[k] - lo[k]) * rng.standard_normal()
                            center[k] = np.clip(center[k], lo, hi)
                    
                    local_X[i] = center
                    local_F[i] = problem(center)
                    used += 1
                    
                    if time.time() > self.deadline:
                        break
                
                # Combine with current population
                if used > 0:
                    R_X = np.vstack([X, local_X[:used - (self.budget - remaining)]])
                    R_F = np.vstack([F, local_F[:used - (self.budget - remaining)]])
                    X, F = self._nsga2_select(R_X, R_F)
        
        return X

    def _calculate_rank_cd(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        cd = np.zeros(n)
        
        # Non-dominated sorting
        remaining = set(range(n))
        current_rank = 0
        
        while remaining:
            front_indices = []
            for i in list(remaining):
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    front_indices.append(i)
                    remaining.remove(i)
            
            if not front_indices:
                break
                
            for i in front_indices:
                ranks[i] = current_rank
            
            if len(front_indices) > 1:
                for m in range(self.n_obj):
                    order = np.argsort(F[front_indices, m])
                    sorted_indices = [front_indices[k] for k in order]
                    
                    cd[sorted_indices[0]] = np.inf
                    cd[sorted_indices[-1]] = np.inf
                    
                    span = F[sorted_indices[-1], m] - F[sorted_indices[0], m]
                    if span > 0:
                        for k in range(1, len(sorted_indices) - 1):
                            idx_curr = sorted_indices[k]
                            idx_prev = sorted_indices[k - 1]
                            idx_next = sorted_indices[k + 1]
                            cd[idx_curr] += (F[idx_next, m] - F[idx_prev, m]) / span
            else:
                for i in front_indices:
                    cd[i] = np.inf
            
            current_rank += 1
            
        return ranks, cd

    def _get_non_dominated_indices(self, F):
        n = len(F)
        non_dominated = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(i)
        return non_dominated

    def _tournament(self, ranks, cd, rng):
        i1 = rng.integers(len(ranks))
        i2 = rng.integers(len(ranks))
        
        if ranks[i1] < ranks[i2]:
            return i1
        elif ranks[i2] < ranks[i1]:
            return i2
        else:
            if cd[i1] > cd[i2]:
                return i1
            elif cd[i2] > cd[i1]:
                return i2
            else:
                return i1

    def _nsga2_select(self, R_X, R_F):
        ranks, cd = self._calculate_rank_cd(R_F)
        
        # Sort by rank asc, then cd desc
        sorted_indices = np.lexsort((-cd, ranks))
        
        selected_idx = sorted_indices[:self.pop_size]
        
        return R_X[selected_idx], R_F[selected_idx]
