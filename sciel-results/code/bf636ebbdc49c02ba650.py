import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        n_doms = np.zeros(n, dtype=int)
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_doms[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_doms[i] += 1
        
        current_front = [i for i in range(n) if n_doms[i] == 0]
        rank = np.full(n, -1, dtype=int)
        front_id = 0
        
        while current_front:
            fronts.append(current_front)
            for i in current_front:
                rank[i] = front_id
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_doms[j] -= 1
                    if n_doms[j] == 0:
                        next_front.append(j)
            current_front = next_front
            front_id += 1
            
        return fronts, rank

    def _crowding_distance_array(self, F, front_indices, n):
        dist = np.zeros(n, dtype=float)
        if len(front_indices) <= 2:
            for i in front_indices:
                dist[i] = np.inf
            return dist
            
        n_obj = self.n_obj
        for m in range(n_obj):
            sorted_front = sorted(front_indices, key=lambda idx: F[idx][m])
            
            dist[sorted_front[0]] = np.inf
            dist[sorted_front[-1]] = np.inf
            
            f_min = F[sorted_front[0]][m]
            f_max = F[sorted_front[-1]][m]
            
            if f_max - f_min == 0:
                continue
                
            for k in range(1, len(sorted_front) - 1):
                idx = sorted_front[k]
                prev_idx = sorted_front[k-1]
                next_idx = sorted_front[k+1]
                dist[idx] += (F[next_idx][m] - F[prev_idx][m]) / (f_max - f_min)
                
        return dist

    def _sbx_crossover(self, x1, x2, lo, hi):
        eta_c = 20.0
        u = np.random.rand(self.dim)
        
        c1 = np.zeros_like(x1)
        c2 = np.zeros_like(x2)
        
        for i in range(self.dim):
            if u[i] <= 0.5:
                alpha = (2 * u[i]) ** (1 / (eta_c + 1))
            else:
                alpha = (2 * (1 - u[i])) ** (1 / (eta_c + 1))
                
            beta = 1 + alpha * (x1[i] - x2[i]) / (hi[i] - lo[i]) if hi[i] != lo[i] else 1
            
            if abs(beta) > 1:
                beta_q = (2 - beta) ** (1 / (eta_c + 1)) if beta > 1 else (-2 * beta) ** (1 / (eta_c + 1))
            else:
                beta_q = beta ** (1 / (eta_c + 1))
                
            c1[i] = 0.5 * ((1 + beta_q) * x1[i] + (1 - beta_q) * x2[i])
            c2[i] = 0.5 * ((1 - beta_q) * x1[i] + (1 + beta_q) * x2[i])
            
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi):
        eta_m = 20.0
        x_mut = x.copy()
        
        for i in range(self.dim):
            if np.random.rand() < 1.0 / self.dim:
                delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i]) if hi[i] != lo[i] else 0
                delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i]) if hi[i] != lo[i] else 0
                
                u = np.random.rand()
                if u < 0.5:
                    xy = 1 + delta1
                    mut = 2 * u
                    delta = (xy ** (1 + 1/eta_m) - 1)
                else:
                    xy = 1 + delta2
                    mut = 2 * (1 - u)
                    delta = 1 - (xy ** (1 + 1/eta_m))
                    
                x_mut[i] += delta * (hi[i] - lo[i])
                
        x_mut = np.clip(x_mut, lo, hi)
        return x_mut

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Smaller population allows more generations and finer-grained search
        # ZDT3 has many local Pareto fronts, so diversity and exploration are key.
        # P=10 gives 19 generations, P=20 gives 9.
        pop_size = 10
        if pop_size >= budget:
            pop_size = max(1, budget // 2)
            
        generations = (budget - pop_size) // pop_size
        
        # Initialize
        X = lo + (hi - lo) * np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used_evals = pop_size
        
        # Main Loop
        for gen in range(generations):
            Q_X = []
            Q_F = []
            
            fronts, ranks = self._non_dominated_sort(F)
            crowds = np.zeros(pop_size, dtype=float)
            for front in fronts:
                temp_dist = self._crowding_distance_array(F, front, pop_size)
                for idx in front:
                    crowds[idx] = temp_dist[idx]
                    
            # Generate offspring
            # We need pop_size offspring.
            # Each crossover produces 2 children.
            # So we need pop_size // 2 crossovers.
            n_crossovers = pop_size // 2
            
            for _ in range(n_crossovers):
                # Tournament selection
                i1 = np.random.randint(pop_size)
                i2 = np.random.randint(pop_size)
                if ranks[i1] < ranks[i2] or (ranks[i1] == ranks[i2] and crowds[i1] > crowds[i2]):
                    p1 = X[i1]
                else:
                    p1 = X[i2]
                    
                i3 = np.random.randint(pop_size)
                i4 = np.random.randint(pop_size)
                if ranks[i3] < ranks[i4] or (ranks[i3] == ranks[i4] and crowds[i3] > crowds[i4]):
                    p2 = X[i3]
                else:
                    p2 = X[i4]
                    
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                
                c1 = self._polynomial_mutation(c1, lo, hi)
                c2 = self._polynomial_mutation(c2, lo, hi)
                
                f1 = problem(c1)
                f2 = problem(c2)
                
                Q_X.append(c1)
                Q_X.append(c2)
                Q_F.append(f1)
                Q_F.append(f2)
                
                used_evals += 2
                if used_evals >= budget:
                    break
                    
            if used_evals >= budget:
                break
            
            # If pop_size is odd, we might have 1 fewer offspring. 
            # But we selected pop_size//2 pairs, so 2*(pop_size//2) offspring.
            # If pop_size is even, this is exactly pop_size.
            # If pop_size is odd, this is pop_size - 1.
            # We can add one more random individual or clone to fill up?
            # Or just proceed with fewer? NSGA-II expects same size.
            # Let's just ensure we have enough.
            
            Q_X = np.array(Q_X)
            Q_F = np.array(Q_F)
            
            # Combine
            R_X = np.vstack((X, Q_X))
            R_F = np.vstack((F, Q_F))
            
            R_fronts, R_ranks = self._non_dominated_sort(R_F)
            
            next_X = []
            next_F = []
            n_selected = 0
            
            for front in R_fronts:
                if n_selected + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(R_X[idx])
                        next_F.append(R_F[idx])
                    n_selected += len(front)
                else:
                    temp_dist = self._crowding_distance_array(R_F, front, len(R_X))
                    front_with_dist = [(idx, temp_dist[idx]) for idx in front]
                    front_with_dist.sort(key=lambda x: x[1], reverse=True)
                    
                    needed = pop_size - n_selected
                    for idx, _ in front_with_dist[:needed]:
                        next_X.append(R_X[idx])
                        next_F.append(R_F[idx])
                    n_selected += needed
                    break
                    
            X = np.array(next_X)
            F = np.array(next_F)
            
        return X
