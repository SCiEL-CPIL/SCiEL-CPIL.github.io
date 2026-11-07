"""Optimizer for DTLZ4 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using NSGA-II with crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """Perform non-dominated sorting. Returns list of fronts (indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = [0] * n  # n_dom[i] = number of points that dominate i
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                # Check if j dominates i
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        # First front: points with no dominators
        for i in range(n):
            if n_dom[i] == 0:
                fronts[0].append(i)
        
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)
        
        # Remove last empty front if any
        if fronts[-1] == []:
            fronts.pop()
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for given indices in F."""
        if len(indices) <= 2:
            return {i: float('inf') for i in indices}
        
        n_obj = self.n_obj
        dist = {i: 0.0 for i in indices}
        
        for m in range(n_obj):
            sorted_idx = sorted(indices, key=lambda i: F[i][m])
            dist[sorted_idx[0]] = float('inf')
            dist[sorted_idx[-1]] = float('inf')
            
            min_val = F[sorted_idx[0]][m]
            max_val = F[sorted_idx[-1]][m]
            range_val = max_val - min_val
            
            if range_val > 1e-15:
                for i in range(1, len(sorted_idx) - 1):
                    dist[sorted_idx[i]] += (F[sorted_idx[i+1]][m] - F[sorted_idx[i-1]][m]) / range_val
        
        return dist

    def _select_parents(self, X, F, fronts):
        """Select parents using tournament selection based on rank and crowding distance."""
        n = len(X)
        # Assign ranks
        rank = np.zeros(n, dtype=int)
        for k, front in enumerate(fronts):
            for idx in front:
                rank[idx] = k
        
        # Compute crowding distance per front
        cd = np.zeros(n)
        for front in fronts:
            if len(front) > 2:
                cd_dict = self._crowding_distance(F, front)
                for idx in front:
                    cd[idx] = cd_dict[idx]
            else:
                for idx in front:
                    cd[idx] = float('inf')
        
        # Tournament selection
        parents = []
        for _ in range(2):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if rank[i] < rank[j]:
                parents.append(i)
            elif rank[j] < rank[i]:
                parents.append(j)
            else:
                if cd[i] > cd[j]:
                    parents.append(i)
                else:
                    parents.append(j)
        return parents

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin within 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        pop_size = min(50, self.budget // 2)
        if pop_size < 4:
            pop_size = 4
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Main loop
        while used < self.budget:
            # Check time
            if time.time() > deadline - 10:
                break
            
            # Non-dominated sorting
            fronts = self._non_dominated_sort(F)
            
            # Select parents
            p1, p2 = self._select_parents(X, F, fronts)
            
            # Crossover (SBX)
            if np.random.random() < 0.9:
                # Simulated Binary Crossover
                eta = 20
                child1 = np.zeros(self.dim)
                child2 = np.zeros(self.dim)
                for d in range(self.dim):
                    x1, x2 = X[p1][d], X[p2][d]
                    if abs(x1 - x2) > 1e-15:
                        u = np.random.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        child1[d] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                        child2[d] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                    else:
                        child1[d] = x1
                        child2[d] = x2
                # Mutation
                for d in range(self.dim):
                    if np.random.random() < 1.0 / self.dim:
                        child1[d] += 0.1 * (hi[d] - lo[d]) * np.random.standard_normal()
                    if np.random.random() < 1.0 / self.dim:
                        child2[d] += 0.1 * (hi[d] - lo[d]) * np.random.standard_normal()
                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)
                
                fc1 = problem(child1)
                used += 1
                fc2 = problem(child2)
                used += 1
                
                # Add to population temporarily for selection
                X_new = np.vstack([X, child1, child2])
                F_new = np.vstack([F, [fc1], [fc2]])
            else:
                # Just mutation
                child = X[p1].copy()
                for d in range(self.dim):
                    if np.random.random() < 1.0 / self.dim:
                        child[d] += 0.1 * (hi[d] - lo[d]) * np.random.standard_normal()
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                
                X_new = np.vstack([X, child])
                F_new = np.vstack([F, [fc]])
            
            # Selection: keep best pop_size from X_new
            fronts_new = self._non_dominated_sort(F_new)
            
            selected = []
            for front in fronts_new:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(selected)
                    if remaining <= 0:
                        break
                    cd_dict = self._crowding_distance(F_new, front)
                    sorted_front = sorted(front, key=lambda i: cd_dict[i], reverse=True)
                    selected.extend(sorted_front[:remaining])
                    break
            
            X = X_new[selected]
            F = F_new[selected]
        
        return X
# EVOLVE-BLOCK-END
