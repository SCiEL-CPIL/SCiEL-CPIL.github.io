import numpy as np
import itertools

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        M = 20
        # Budget check: 200 evals. Init: 20. Remaining: 180.
        # Generations: 180 / 20 = 9 generations.
        
        # 1. Initialize Population
        X = lo + (hi - lo) * np.random.random((M, self.dim))
        F = np.array([problem(x) for x in X])
        
        used = M
        
        # Helper functions for Pareto dominance and non-dominated sorting
        
        def dominates(a, b):
            # a dominates b if a is better or equal in all objectives and strictly better in at least one
            # Minimizing objectives
            return np.all(a <= b) and np.any(a < b)
        
        def non_dominated_sort(F):
            # Returns list of fronts, each front is a list of indices
            n = len(F)
            fronts = []
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            n_dominated = [0] * n       # n_dominated[i] = number of solutions dominating i
            rank = [0] * n              # rank[i] = front index
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_dominated[i] += 1
            
            # First front
            current_front = []
            for i in range(n):
                if n_dominated[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
            
            fronts.append(current_front)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            
            return fronts
        
        def crowding_distance(F, indices):
            # Calculate crowding distance for a set of indices in F
            # Returns a dictionary of index -> distance
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            dist = {i: 0.0 for i in indices}
            for m in range(self.n_obj):
                # Sort indices by m-th objective
                sorted_indices = sorted(indices, key=lambda i: F[i][m])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0]][m]
                f_max = F[sorted_indices[-1]][m]
                
                if f_max == f_min:
                    continue
                    
                for k in range(1, len(sorted_indices) - 1):
                    i = sorted_indices[k]
                    dist[i] += (F[sorted_indices[k+1]][m] - F[sorted_indices[k-1]][m]) / (f_max - f_min)
            
            return dist
        
        # Main Evolution Loop
        while used < self.budget:
            # Generate offspring
            offspring_X = np.zeros((M, self.dim))
            offspring_F = np.zeros((M, self.n_obj))
            
            for i in range(M):
                if used >= self.budget:
                    break
                    
                # DE/rand/1/bin mutation
                # Select 3 distinct random parents
                parents = np.random.choice(M, 3, replace=False)
                p1, p2, p3 = parents[0], parents[1], parents[2]
                
                F_factor = 0.8
                C_rate = 0.5
                
                v = X[p1] + F_factor * (X[p2] - X[p3])
                
                # Binary crossover
                cross_mask = np.random.random(self.dim) < C_rate
                # Ensure at least one gene is mutated
                if not np.any(cross_mask):
                    j = np.random.randint(self.dim)
                    cross_mask[j] = True
                    
                child = np.where(cross_mask, v, X[i])
                child = np.clip(child, lo, hi)
                
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1
            
            # Combine parents and offspring
            # We only care about the best M from the combined 2M
            # But we must respect the budget. We have evaluated M offspring.
            
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sorting
            fronts = non_dominated_sort(combined_F)
            
            new_X = []
            new_F = []
            
            for front in fronts:
                if len(new_X) + len(front) <= M:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front to fill remaining slots
                    remaining = M - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
        return F
