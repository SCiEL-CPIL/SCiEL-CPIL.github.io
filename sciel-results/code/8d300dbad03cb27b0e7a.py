import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Parameters
        # Budget is 200. We want to use it efficiently.
        # A population of 20 with 9 generations (180 evals) + 20 init = 200.
        pop_size = 20
        n_gen = (self.budget - pop_size) // pop_size
        
        # 1. Initialization
        # Use Latin Hypercube Sampling (LHS) for better coverage
        X = np.zeros((pop_size, self.dim))
        for j in range(self.dim):
            # LHS: divide [0,1] into pop_size bins, pick one random in each, shuffle
            u = np.random.rand(pop_size)
            # Map to bins: (u + i) / pop_size, then shuffle
            # Actually, standard LHS:
            # indices = np.random.permutation(pop_size)
            # X[:, j] = lo[j] + (hi[j] - lo[j]) * (indices + np.random.rand(pop_size)) / pop_size
            indices = np.random.permutation(pop_size)
            X[:, j] = lo[j] + range_[j] * (indices + np.random.rand(pop_size)) / pop_size
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Check if point is within bounds (problem clips, but we can be strict)
        # Problem clips, so X is always valid.
        
        # Helper: Dominance
        def dominates(a, b):
            # a dominates b if a <= b and a != b
            return np.all(a <= b) and np.any(a < b)
        
        # Helper: Non-dominated sort (fast, vectorized where possible)
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            fronts = [[]]
            
            for p in range(n):
                for q in range(n):
                    if p != q:
                        if dominates(F[p], F[q]):
                            S[p].append(q)
                        elif dominates(F[q], F[p]):
                            n_count[p] += 1
                if n_count[p] == 0:
                    fronts[0].append(p)
                    
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            return fronts[:-1]
        
        # Helper: Crowding distance
        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            dist = [0.0] * n
            for m in range(self.n_obj):
                sorted_indices = sorted(indices, key=lambda i: F[i, m])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                f_min = F[sorted_indices[0], m]
                f_max = F[sorted_indices[-1], m]
                if f_max == f_min:
                    continue
                for k in range(1, n - 1):
                    dist[sorted_indices[k]] += (F[sorted_indices[k+1], m] - F[sorted_indices[k-1], m]) / (f_max - f_min)
            return dist
        
        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta=15):
            x = np.zeros_like(x1)
            for j in range(self.dim):
                if np.random.rand() < 0.5:
                    if abs(x1[j] - x2[j]) < 1e-14:
                        x[j] = x1[j]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        x[j] = 0.5 * ((1 + beta) * x1[j] + (1 - beta) * x2[j])
                        x[j] = np.clip(x[j], lo[j], hi[j])
                else:
                    x[j] = x1[j]
            return x
        
        # Helper: Polynomial Mutation
        def poly_mutation(x, eta=20):
            x = x.copy()
            for j in range(self.dim):
                if np.random.rand() < 1.0 / self.dim:
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    x[j] = x[j] + delta * range_[j]
                    x[j] = np.clip(x[j], lo[j], hi[j])
            return x
        
        # Evolutionary Loop
        for gen in range(n_gen):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
                
            # Non-dominated sort
            fronts = non_dominated_sort(F)
            
            # Select parents using tournament selection
            children_X = []
            children_F = []
            
            # Generate pop_size children
            for _ in range(pop_size):
                if used >= self.budget:
                    break
                
                # Tournament for parent 1
                p1_idx = np.random.randint(0, pop_size)
                p2_idx = np.random.randint(0, pop_size)
                
                # Determine ranks
                rank1 = next((i for i, f in enumerate(fronts) if p1_idx in f), len(fronts))
                rank2 = next((i for i, f in enumerate(fronts) if p2_idx in f), len(fronts))
                
                if rank1 < rank2:
                    p1 = p1_idx
                elif rank2 < rank1:
                    p1 = p2_idx
                else:
                    # Crowding distance
                    cd = crowding_distance(F, list(range(pop_size)))
                    p1 = p1_idx if cd[p1_idx] > cd[p2_idx] else p2_idx
                    
                # Tournament for parent 2
                p3_idx = np.random.randint(0, pop_size)
                p4_idx = np.random.randint(0, pop_size)
                
                rank3 = next((i for i, f in enumerate(fronts) if p3_idx in f), len(fronts))
                rank4 = next((i for i, f in enumerate(fronts) if p4_idx in f), len(fronts))
                
                if rank3 < rank4:
                    p2 = p3_idx
                elif rank4 < rank3:
                    p2 = p4_idx
                else:
                    cd = crowding_distance(F, list(range(pop_size)))
                    p2 = p3_idx if cd[p3_idx] > cd[p4_idx] else p4_idx
                
                # Crossover and Mutation
                child_x = sbx_crossover(X[p1], X[p2])
                child_x = poly_mutation(child_x)
                
                child_f = problem(child_x)
                used += 1
                
                children_X.append(child_x)
                children_F.append(child_f)
                
            if not children_X:
                break
                
            # Combine
            X = np.vstack([X, np.array(children_X)])
            F = np.vstack([F, np.array(children_F)])
            
            # Selection
            fronts = non_dominated_sort(F)
            new_X = []
            new_F = []
            
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X[idx])
                        new_F.append(F[idx])
                else:
                    # Crowding distance for this front
                    cd = crowding_distance(F, front)
                    # Sort front by crowding distance descending
                    pairs = [(idx, cd[front.index(idx)]) for idx in front]
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    needed = pop_size - len(new_X)
                    for idx, _ in pairs[:needed]:
                        new_X.append(X[idx])
                        new_F.append(F[idx])
                    break
                    
            X = np.array(new_X)
            F = np.array(new_F)
            
        return X
