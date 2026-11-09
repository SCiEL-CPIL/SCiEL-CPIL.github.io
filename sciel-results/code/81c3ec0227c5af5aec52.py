import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use np.random to respect the seed set by the evaluator
        # pop_size = 10 allows 20 generations with budget 200
        pop_size = 10
        
        # Initialize
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        eta_c = 20.0
        pc = 0.9
        sigma_scale = 0.05 # Reduced from 0.1
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            Q = np.empty_like(X)
            for i in range(pop_size):
                # Tournament Selection
                p1 = self._tournament_select(X, F)
                p2 = self._tournament_select(X, F)
                
                # Crossover (SBX)
                if np.random.random() < pc:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Gaussian Mutation
                c1 = c1 + sigma_scale * (hi - lo) * np.random.standard_normal(self.dim)
                c2 = c2 + sigma_scale * (hi - lo) * np.random.standard_normal(self.dim)
                
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                
                Q[i] = c1
                
            F_q = np.array([problem(x) for x in Q])
            used += pop_size
            
            P = np.vstack([X, Q])
            F_p = np.vstack([F, F_q])
            
            X, F = self._nsga2_select(P, F_p, pop_size)
            
        return X

    def _tournament_select(self, X, F, k=2):
        idxs = np.random.choice(len(F), size=k, replace=False)
        best_idx = idxs[0]
        for idx in idxs[1:]:
            if self._dominates(F[idx], F[best_idx]):
                best_idx = idx
        return X[best_idx].copy()

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _nsga2_select(self, P, F_p, target_size):
        fronts = self._non_dominated_sort(F_p)
        
        next_X = []
        next_F = []
        
        for front in fronts:
            if len(next_X) + len(front) <= target_size:
                next_X.extend(P[front])
                next_F.extend(F_p[front])
            else:
                F_front = F_p[front]
                cd = self._crowding_distance(F_front)
                sorted_indices = np.argsort(-cd)
                needed = target_size - len(next_X)
                for idx in sorted_indices[:needed]:
                    global_idx = front[idx]
                    next_X.append(P[global_idx])
                    next_F.append(F_p[global_idx])
                break
                
        return np.array(next_X), np.array(next_F)

    def _non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        n_dominating = [0] * n
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    n_dominating[i] += 1
                    
        current_front = [i for i in range(n) if n_dominating[i] == 0]
        
        k = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dominating[j] -= 1
                    if n_dominating[j] == 0:
                        next_front.append(j)
            current_front = next_front
            k += 1
            
        return fronts

    def _crowding_distance(self, F):
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
            
        cd = np.zeros(n)
        m = F.shape[1]
        
        for k in range(m):
            sorted_idx = np.argsort(F[:, k])
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            
            f_range = F[sorted_idx[-1], k] - F[sorted_idx[0], k]
            if f_range == 0:
                continue
                
            for i in range(1, n - 1):
                cd[sorted_idx[i]] += (F[sorted_idx[i+1], k] - F[sorted_idx[i-1], k]) / f_range
                
        return cd

    def _sbx_crossover(self, p1, p2, lo, hi, eta):
        c1 = np.empty_like(p1)
        c2 = np.empty_like(p2)
        
        for i in range(len(p1)):
            if np.random.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if 2 * np.random.random() <= (1 / (eta + 1)):
                        beta = (2 * np.random.random()) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - np.random.random()))) ** (1 / (eta + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                else:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                c1[i] = p2[i]
                c2[i] = p1[i]
                
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        
        return c1, c2
