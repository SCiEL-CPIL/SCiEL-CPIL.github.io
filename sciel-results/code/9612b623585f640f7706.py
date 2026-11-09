import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = 20
        eta_c = 15.0
        eta_m = 20.0
        pc = 0.9
        
        # 1. Initialization
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Pre-compute ranks and CD for initial population
        fronts, ranks = self._fast_non_dominated_sort(F)
        cd = self._calc_crowding(F, fronts)
        
        while used + N <= self.budget:
            # 2. Generate Offspring
            Q_X = []
            for i in range(N):
                # Tournament Selection
                p1_idx = np.random.randint(N)
                p2_idx = np.random.randint(N)
                
                # Compare based on rank and crowding distance
                if ranks[p1_idx] < ranks[p2_idx]:
                    p_idx = p1_idx
                elif ranks[p2_idx] < ranks[p1_idx]:
                    p_idx = p2_idx
                else:
                    if cd[p1_idx] > cd[p2_idx]:
                        p_idx = p1_idx
                    else:
                        p_idx = p2_idx
                
                p1 = X[p_idx]
                
                # Select second parent
                p1_idx2 = np.random.randint(N)
                p2_idx2 = np.random.randint(N)
                if ranks[p1_idx2] < ranks[p2_idx2]:
                    p_idx2 = p1_idx2
                elif ranks[p2_idx2] < ranks[p1_idx2]:
                    p_idx2 = p2_idx2
                else:
                    if cd[p1_idx2] > cd[p2_idx2]:
                        p_idx2 = p1_idx2
                    else:
                        p_idx2 = p2_idx2
                        
                p2 = X[p_idx2]
                
                if np.random.rand() < pc:
                    c1, _ = self._sbx(p1, p2, lo, hi, eta_c)
                else:
                    c1 = p1.copy()
                
                c1 = self._mutate(c1, lo, hi, eta_m)
                Q_X.append(c1)
            
            Q_F = np.array([problem(x) for x in Q_X])
            used += N
            
            # 3. Combine
            R_X = np.vstack([X, Q_X])
            R_F = np.vstack([F, Q_F])
            
            # 4. Non-Dominated Sort
            fronts_R, ranks_R = self._fast_non_dominated_sort(R_F)
            cd_R = self._calc_crowding(R_F, fronts_R)
            
            # 5. Select Next Population
            new_X = []
            new_F = []
            current_size = 0
            for front in fronts_R:
                if current_size + len(front) <= N:
                    for idx in front:
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                        current_size += 1
                else:
                    remaining = N - current_size
                    # Sort front by crowding distance descending
                    front_cd = cd_R[front]
                    sorted_indices = np.argsort(-front_cd)
                    for i in range(remaining):
                        idx = front[sorted_indices[i]]
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
            # Update ranks and CD for next iteration
            fronts, ranks = self._fast_non_dominated_sort(F)
            cd = self._calc_crowding(F, fronts)
            
        return X

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        S = [set() for _ in range(n)]
        n_d = np.zeros(n, dtype=int)
        rank = np.zeros(n, dtype=int)
        fronts = []
        
        for p in range(n):
            for q in range(n):
                if p == q: continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].add(q)
                    n_d[q] += 1
        
        front0 = [p for p in range(n) if n_d[p] == 0]
        if not front0:
             # Fallback: all in one front
             return [list(range(n))], np.zeros(n, dtype=int)
             
        rank[front0] = 0
        fronts.append(front0)
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n_d[q] -= 1
                    if n_d[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
                i += 1
            else:
                break
        return fronts, rank

    def _calc_crowding(self, F, fronts):
        cd = np.zeros(len(F))
        for front in fronts:
            if len(front) <= 2:
                for idx in front:
                    cd[idx] = np.inf
                continue
            m = len(front)
            front_F = F[front]
            k = self.n_obj
            front_cd = np.zeros(m)
            
            for j in range(k):
                obj = front_F[:, j]
                sorted_indices = np.argsort(obj)
                front_cd[sorted_indices[0]] = np.inf
                front_cd[sorted_indices[-1]] = np.inf
                obj_range = obj[sorted_indices[-1]] - obj[sorted_indices[0]]
                if obj_range == 0:
                    continue
                for i in range(1, m - 1):
                    front_cd[sorted_indices[i]] += (obj[sorted_indices[i+1]] - obj[sorted_indices[i-1]]) / obj_range
            
            for i, idx in enumerate(front):
                cd[idx] = front_cd[i]
        return cd

    def _sbx(self, x1, x2, lo, hi, eta):
        c1 = x1.copy()
        c2 = x2.copy()
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    if np.random.rand() < 0.5:
                        beta = (2 * np.random.rand()) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * np.random.rand())) ** (1.0 / (eta + 1.0))
                    
                    c1[i] = 0.5 * (1 + beta) * x1[i] + 0.5 * (1 - beta) * x2[i]
                    c2[i] = 0.5 * (1 - beta) * x1[i] + 0.5 * (1 + beta) * x2[i]
        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _mutate(self, x, lo, hi, eta):
        y = x.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                delta1 = (y[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - y[i]) / (hi[i] - lo[i])
                rand_val = np.random.rand()
                if rand_val < 0.5:
                    xy = 1.0 + (1.0 - delta1) * rand_val
                    deltaq = (2.0 * xy) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    xy = 1.0 + delta2 * rand_val
                    deltaq = 1.0 - (2.0 * xy) ** (1.0 / (eta + 1.0))
                
                y[i] = y[i] + deltaq * (hi[i] - lo[i])
        
        return np.clip(y, lo, hi)
