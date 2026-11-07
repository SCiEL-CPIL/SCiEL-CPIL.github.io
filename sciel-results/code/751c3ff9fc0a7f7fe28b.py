import numpy as np
import time

class Optimizer:
    """
    Optimizer for RE37 multi-objective problem.
    Uses an NSGA-II-like approach with crowded distance sorting.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        bounds_range = hi - lo
        
        # We'll keep a population of non-dominated points
        # and use tournament selection based on rank and crowding distance
        
        pop_size = min(40, self.budget // 5)
        if pop_size < 10:
            pop_size = 10
            
        # Initialize population
        X = lo + bounds_range * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # Helper functions
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            front = [[]]
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if dominates(F[p], F[q]):
                        S[p].append(q)
                    elif dominates(F[q], F[p]):
                        n_count[p] += 1
                if n_count[p] == 0:
                    front[0].append(p)
                    
            i = 0
            while len(front[i]) > 0:
                next_front = []
                for p in front[i]:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            next_front.append(q)
                i += 1
                if next_front:
                    front.append(next_front)
                    
            return front
        
        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            
            cd = [0.0] * n
            m = len(F[0])
            
            for k in range(m):
                vals = [F[idx][k] for idx in front_indices]
                min_val = min(vals)
                max_val = max(vals)
                if max_val - min_val < 1e-15:
                    continue
                    
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                
                for i in range(1, n - 1):
                    prev_val = vals[sorted_idx[i-1]]
                    next_val = vals[sorted_idx[i+1]]
                    cd[sorted_idx[i]] += (next_val - prev_val) / (max_val - min_val)
                    
            return cd
        
        def tournament_select(pop_F, ranks, cd_values, k=2):
            candidates = [np.random.randint(len(pop_F)) for _ in range(k)]
            best = candidates[0]
            for c in candidates[1:]:
                if ranks[c] < ranks[best]:
                    best = c
                elif ranks[c] == ranks[best]:
                    if cd_values[c] > cd_values[best]:
                        best = c
            return best
        
        # Main loop
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Compute ranks and crowding distance
            fronts = fast_non_dominated_sort(F)
            ranks = [0] * len(F)
            cd_values = [0.0] * len(F)
            
            for i, front in enumerate(fronts):
                cd = crowding_distance(F, front)
                for j, idx in enumerate(front):
                    ranks[idx] = i
                    cd_values[idx] = cd[j]
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size and evals < self.budget:
                if time.time() > deadline:
                    break
                    
                p1 = tournament_select(F, ranks, cd_values)
                p2 = tournament_select(F, ranks, cd_values)
                
                # Simulated binary crossover (SBX)
                if np.random.random() < 0.9:
                    x1 = X[p1].copy()
                    x2 = X[p2].copy()
                    
                    for j in range(self.dim):
                        if np.random.random() < 0.5:
                            if x1[j] != x2[j]:
                                if np.random.random() < 0.5:
                                    beta = 1.0 / (1.0 + 2.0 * (x1[j] - x2[j]))
                                else:
                                    beta = 1.0 / (1.0 + 2.0 * (x2[j] - x1[j]))
                                
                                beta_q = ((beta + 1) / 2) ** 5
                                x1[j] = 0.5 * ((1 + beta) * x1[j] + (1 - beta) * x2[j])
                                x2[j] = 0.5 * ((1 - beta_q) * x1[j] + (1 + beta_q) * x2[j])
                
                # Polynomial mutation
                for j in range(self.dim):
                    if np.random.random() < 0.8:
                        x1[j] += 0.1 * (hi[j] - lo[j]) * np.random.standard_normal()
                
                x1 = np.clip(x1, lo, hi)
                
                f1 = np.array(problem(x1))
                evals += 1
                offspring_X.append(x1)
                offspring_F.append(f1)
                
                # Also evaluate second child if budget allows
                if len(offspring_X) < pop_size and evals < self.budget:
                    for j in range(self.dim):
                        if np.random.random() < 0.8:
                            x2[j] += 0.1 * (hi[j] - lo[j]) * np.random.standard_normal()
                    
                    x2 = np.clip(x2, lo, hi)
                    
                    f2 = np.array(problem(x2))
                    evals += 1
                    offspring_X.append(x2)
                    offspring_F.append(f2)
            
            if not offspring_X:
                break
                
            # Combine and select
            X = np.vstack([X, np.array(offspring_X)])
            F = np.vstack([F, np.array(offspring_F)])
            
            # Truncate to pop_size using non-dominated sorting
            fronts = fast_non_dominated_sort(F)
            new_indices = []
            
            for front in fronts:
                if len(new_indices) + len(front) <= pop_size:
                    new_indices.extend(front)
                else:
                    # Need to select from this front based on crowding distance
                    cd = crowding_distance(F, front)
                    sorted_by_cd = np.argsort(cd)[::-1]
                    needed = pop_size - len(new_indices)
                    for i in range(min(needed, len(sorted_by_cd))):
                        new_indices.append(front[sorted_by_cd[i]])
                    break
            
            X = X[new_indices]
            F = F[new_indices]
            
        # Return all non-dominated points found
        fronts = fast_non_dominated_sort(F)
        final_indices = []
        for front in fronts:
            final_indices.extend(front)
        
        return [X[i] for i in final_indices]
