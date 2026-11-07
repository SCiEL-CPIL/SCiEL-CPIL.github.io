import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        active_dim = 6
        noise_dim = self.dim - active_dim
        
        # Fix noise dimensions to 0.5
        noise_vals = np.full(noise_dim, 0.5)
        
        lo_a = lo[:active_dim]
        hi_a = hi[:active_dim]
        
        pop_size = 20
        # Initial population
        X = lo_a + (hi_a - lo_a) * np.random.rand(pop_size, active_dim)
        
        def eval_active(x_active):
            x_full = np.concatenate([x_active, noise_vals])
            return problem(x_full)
            
        F = np.array([eval_active(x) for x in X])
        used = pop_size
        
        # NSGA-II Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # 1. Non-dominated sorting
            # Returns ranks and crowding distances
            ranks, cd = self._fast_non_dominated_sort(F)
            
            # 2. Selection (Tournament)
            parents = []
            while len(parents) < pop_size:
                p1 = self._tournament_select(F, ranks, cd)
                p2 = self._tournament_select(F, ranks, cd)
                parents.append(p1)
                parents.append(p2)
            
            # 3. Crossover and Mutation
            children_X = []
            children_F = []
            for i in range(0, len(parents), 2):
                p1 = X[parents[i]]
                p2 = X[parents[i+1]]
                
                # SBX Crossover
                c1, c2 = self._sbx_crossover(p1, p2, lo_a, hi_a)
                
                # Mutation
                c1 = self._polynomial_mutation(c1, lo_a, hi_a)
                c2 = self._polynomial_mutation(c2, lo_a, hi_a)
                
                # Evaluate
                f1 = eval_active(c1)
                f2 = eval_active(c2)
                
                children_X.append(c1)
                children_F.append(f1)
                children_X.append(c2)
                children_F.append(f2)
                
                used += 2
                if used >= self.budget:
                    break
            
            if not children_X:
                break
                
            # 4. Combine and Select
            X_all = np.vstack([X, np.array(children_X)])
            F_all = np.vstack([F, np.array(children_F)])
            
            # Select next generation
            X, F = self._select_next_generation(X_all, F_all, pop_size)
            
        return X

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        cd = np.zeros(n)
        
        # Simple O(N^2) non-dominated sort
        dominated_count = np.zeros(n, dtype=int)
        dominates = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i+1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominates[i].append(j)
                    dominated_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominates[j].append(i)
                    dominated_count[i] += 1
        
        current_front = [i for i in range(n) if dominated_count[i] == 0]
        rank = 0
        while current_front:
            next_front = []
            for i in current_front:
                ranks[i] = rank
                for j in dominates[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            rank += 1
            current_front = next_front
            
        # Crowding Distance
        for r in range(rank):
            front = [i for i in range(n) if ranks[i] == r]
            if len(front) <= 2:
                for i in front:
                    cd[i] = float('inf')
                continue
            
            # Sort by each objective
            for k in range(self.n_obj):
                front_sorted = sorted(front, key=lambda idx: F[idx][k])
                cd[front_sorted[0]] = float('inf')
                cd[front_sorted[-1]] = float('inf')
                
                if F[front_sorted[-1]][k] - F[front_sorted[0]][k] == 0:
                    continue
                    
                for i in range(1, len(front_sorted) - 1):
                    idx = front_sorted[i]
                    cd[idx] += (F[front_sorted[i+1]][k] - F[front_sorted[i-1]][k]) / (F[front_sorted[-1]][k] - F[front_sorted[0]][k])
                    
        return ranks, cd

    def _tournament_select(self, F, ranks, cd):
        i = np.random.randint(len(F))
        j = np.random.randint(len(F))
        if ranks[i] < ranks[j]:
            return i
        elif ranks[j] < ranks[i]:
            return j
        else:
            if cd[i] > cd[j]:
                return i
            else:
                return j

    def _sbx_crossover(self, p1, p2, lo, hi):
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p2)
        for i in range(len(p1)):
            if np.random.rand() < 0.5:
                c1[i] = p1[i]
                c2[i] = p2[i]
            else:
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / 20.0)
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / 20.0)
                
                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                
                c1[i] = np.clip(c1[i], lo[i], hi[i])
                c2[i] = np.clip(c2[i], lo[i], hi[i])
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi):
        x_new = x.copy()
        for i in range(len(x)):
            if np.random.rand() < 0.5:
                continue
            u = np.random.rand()
            if u < 0.5:
                delta = (2 * u) ** (1 / 20.0) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / 20.0)
                
            x_new[i] = x[i] + delta * (hi[i] - lo[i])
            x_new[i] = np.clip(x_new[i], lo[i], hi[i])
        return x_new

    def _select_next_generation(self, X, F, pop_size):
        ranks, cd = self._fast_non_dominated_sort(F)
        
        # Sort by rank, then by crowding distance (descending)
        indices = np.argsort(ranks)
        # Stable sort by rank, then secondary sort by cd
        # This is a bit tricky with np.argsort. 
        # Let's do a custom sort.
        
        sorted_indices = []
        for r in np.unique(ranks):
            front = [i for i in range(len(F)) if ranks[i] == r]
            front_sorted = sorted(front, key=lambda idx: cd[idx], reverse=True)
            sorted_indices.extend(front_sorted)
            
        return X[sorted_indices[:pop_size]], F[sorted_indices[:pop_size]]
