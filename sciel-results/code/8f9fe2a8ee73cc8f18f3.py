import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size: smaller for more generations
        N = 10
        
        if self.budget < N:
            X = lo + (hi - lo) * np.random.random((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return X

        # Initialize
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # DE parameters
        F_scale = 0.8  # Differential weight
        CR = 0.5       # Crossover rate
        
        # Archive: store non-dominated solutions
        archive_X = list(X)
        archive_F = list(F)
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring_X = np.empty_like(X)
            offspring_F = np.empty_like(F)
            
            for i in range(N):
                if time.time() > deadline or used >= self.budget:
                    break
                
                # DE/rand/1/bin mutation
                idx1, idx2, idx3 = np.random.choice(N, 3, replace=False)
                v = X[idx1] + F_scale * (X[idx2] - X[idx3])
                
                # Binomial crossover
                j_rand = np.random.randint(0, dim)
                mask = np.random.random(dim) < CR
                mask[j_rand] = True
                trial = np.where(mask, v, X[i])
                
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
                
                offspring_X[i] = trial
                offspring_F[i] = problem(trial)
                used += 1
                
                # Update archive: if trial dominates any archive member, remove dominated
                new_F = offspring_F[i]
                new_X = trial.copy()
                
                # Check if new point is dominated by archive
                is_dominated = False
                for af in archive_F:
                    if self._dominates(af, new_F):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    # Remove archive members dominated by new point
                    new_archive_F = []
                    new_archive_X = []
                    for j in range(len(archive_F)):
                        if self._dominates(new_F, archive_F[j]):
                            continue
                        new_archive_F.append(archive_F[j])
                        new_archive_X.append(archive_X[j])
                    archive_F = new_archive_F
                    archive_X = new_archive_X
                    archive_F.append(new_F)
                    archive_X.append(new_X)
            
            if used >= self.budget:
                break
            
            # Combine parent and offspring, select best N
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Select N non-dominated with diversity
            X, F = self._select_diverse(X_comb, F_comb, N)

        return X

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _select_diverse(self, X, F, N):
        n = len(X)
        
        # Find non-dominated set
        nondom = []
        for i in range(n):
            is_dom = False
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[j], F[i]):
                    is_dom = True
                    break
            if not is_dom:
                nondom.append(i)
        
        if len(nondom) <= N:
            # Take all non-dominated, fill rest with best dominated
            selected = list(nondom)
            if len(selected) < N:
                # Add remaining dominated by distance to reference
                remaining = [i for i in range(n) if i not in selected]
                # Sort by sum of objectives (simple heuristic)
                remaining.sort(key=lambda i: np.sum(F[i]))
                selected.extend(remaining[:N - len(selected)])
            return X[selected], F[selected]
        
        # Too many non-dominated: select by crowding distance
        cd = self._crowding_distance(F, nondom)
        sorted_nondom = sorted(nondom, key=lambda i: cd[i], reverse=True)
        selected = sorted_nondom[:N]
        return X[selected], F[selected]

    def _crowding_distance(self, F, indices):
        n = len(indices)
        cd = np.zeros(n)
        if n <= 2:
            cd[:] = float('inf')
            return cd
        
        for k in range(self.n_obj):
            sorted_idx = sorted(range(n), key=lambda i: F[indices[i]][k])
            cd[sorted_idx[0]] = float('inf')
            cd[sorted_idx[-1]] = float('inf')
            
            min_val = F[indices[sorted_idx[0]]][k]
            max_val = F[indices[sorted_idx[-1]]][k]
            range_val = max_val - min_val
            if range_val == 0:
                continue
            
            for j in range(1, n - 1):
                prev = F[indices[sorted_idx[j-1]]][k]
                nxt = F[indices[sorted_idx[j+1]]][k]
                cd[sorted_idx[j]] += (nxt - prev) / range_val
        
        return cd
