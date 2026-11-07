"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size: use a good fraction of budget
        pop_size = min(50, max(20, self.budget // 4))
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Non-dominated sort helper
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            rank = [0] * n
            front = [0]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    rank[i] = 0
                    front[0] += 1
            
            k = 0
            while front[k] > 0:
                next_front = []
                for i in range(n):
                    if rank[i] == k:
                        for j in S[i]:
                            n_count[j] -= 1
                            if n_count[j] == 0:
                                rank[j] = k + 1
                                next_front.append(j)
                k += 1
                if k < len(front):
                    front[k] = len(next_front)
                else:
                    front.append(len(next_front))
                if len(next_front) == 0:
                    break
            
            return rank, S
        
        # Crowding distance helper
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return [float('inf')] * len(indices)
            m = self.n_obj
            dist = [0.0] * len(indices)
            for i in range(m):
                sorted_idx = sorted(indices, key=lambda idx: F[idx, i])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                fmin = F[sorted_idx[0], i]
                fmax = F[sorted_idx[-1], i]
                if fmax == fmin:
                    continue
                for j in range(1, len(sorted_idx) - 1):
                    idx = sorted_idx[j]
                    pos = indices.index(idx)
                    dist[pos] += (F[sorted_idx[j + 1], i] - F[sorted_idx[j - 1], i]) / (fmax - fmin)
            return dist
        
        # Selection helper
        def select_parents(F, rank, S):
            n = len(F)
            # Get all fronts
            fronts = {}
            for i in range(n):
                r = rank[i]
                if r not in fronts:
                    fronts[r] = []
                fronts[r].append(i)
            
            parents = []
            for r in sorted(fronts.keys()):
                front = fronts[r]
                if len(parents) + len(front) <= n // 2:
                    parents.extend(front)
                else:
                    # Need crowding distance
                    cd = crowding_distance(F, front)
                    sorted_front = sorted(front, key=lambda idx: cd[front.index(idx)], reverse=True)
                    need = n // 2 - len(parents)
                    parents.extend(sorted_front[:need])
                    break
            
            return parents[:n // 2]
        
        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sort
            rank, S = non_dominated_sort(F)
            
            # Select parents
            parents = select_parents(F, rank, S)
            
            # Create children
            children_X = []
            children_F = []
            num_children = min(pop_size, self.budget - used)
            
            for _ in range(num_children):
                # Pick two parents via tournament
                p1_idx = parents[rng.integers(len(parents))]
                p2_idx = parents[rng.integers(len(parents))]
                
                # Crossover (SBX-like, simplified)
                if rng.random() < 0.9:
                    # Blend crossover
                    alpha = rng.random()
                    child = (1 - alpha) * X[p1_idx] + alpha * X[p2_idx]
                else:
                    child = X[p1_idx].copy()
                
                # Mutation (Gaussian)
                if rng.random() < 0.9:
                    mutation_scale = 0.1 * (hi - lo)
                    child = child + mutation_scale * rng.standard_normal(self.dim)
                
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                
                children_X.append(child)
                children_F.append(fc)
                
                if used >= self.budget:
                    break
            
            if not children_X:
                break
            
            children_X = np.array(children_X)
            children_F = np.array(children_F)
            
            # Combine parents and children
            combined_X = np.vstack([X, children_X])
            combined_F = np.vstack([F, children_F])
            
            # Non-dominated sort on combined
            c_rank, c_S = non_dominated_sort(combined_F)
            
            # Select new population
            new_X = []
            new_F = []
            fronts = {}
            for i in range(len(combined_F)):
                r = c_rank[i]
                if r not in fronts:
                    fronts[r] = []
                fronts[r].append(i)
            
            for r in sorted(fronts.keys()):
                front = fronts[r]
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda idx: cd[front.index(idx)], reverse=True)
                    need = pop_size - len(new_X)
                    for idx in sorted_front[:need]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
        
        return X
# EVOLVE-BLOCK-END
