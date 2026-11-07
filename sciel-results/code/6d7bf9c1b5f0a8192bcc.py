import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)
        
        pop_size = 20
        # Initialize
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        
        archive_x = []
        archive_f = []
        self._update_archive(archive_x, archive_f, X, F)
        
        evals_used = pop_size
        gen = 0
        
        while evals_used < self.budget:
            if time.time() > deadline:
                break
            
            # Create offspring
            offspring_X = np.zeros((pop_size, self.dim))
            offspring_F = np.zeros((pop_size, self.n_obj))
            
            for i in range(pop_size):
                # Tournament selection for parents
                p1 = self._tournament_select(X, F, rng, 2)
                p2 = self._tournament_select(X, F, rng, 2)
                
                # Crossover
                child = self._crossover(p1, p2, rng)
                
                # Mutation
                child = self._mutate(child, lo, hi, rng)
                
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                evals_used += 1
            
            # Combine parents and offspring
            X_all = np.vstack((X, offspring_X))
            F_all = np.vstack((F, offspring_F))
            
            # Selection
            X, F = self._select_next_gen(X_all, F_all, pop_size, rng)
            
            self._update_archive(archive_x, archive_f, X, F)
            
            gen += 1
            
        return archive_x

    def _update_archive(self, archive_x, archive_f, X, F):
        for i in range(len(X)):
            self._add_to_archive(archive_x, archive_f, X[i], F[i])

    def _add_to_archive(self, archive_x, archive_f, x, f):
        if len(archive_f) == 0:
            archive_x.append(x)
            archive_f.append(f)
            return

        F_arr = np.array(archive_f)
        
        # Check if new point is dominated
        dominated = False
        for i in range(len(archive_f)):
            if np.all(F_arr[i] <= f) and np.any(F_arr[i] < f):
                dominated = True
                break
        
        if dominated:
            return
        
        # Remove points dominated by the new point
        new_archive_x = []
        new_archive_f = []
        for i in range(len(archive_f)):
            if np.all(f <= F_arr[i]) and np.any(f < F_arr[i]):
                continue
            new_archive_x.append(archive_x[i])
            new_archive_f.append(archive_f[i])
            
        new_archive_x.append(x)
        new_archive_f.append(f)
        
        archive_x.clear()
        archive_f.clear()
        archive_x.extend(new_archive_x)
        archive_f.extend(new_archive_f)

    def _non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] contains indices dominated by i
        n_i = np.zeros(n, dtype=int)  # Number of points dominating i
        rank = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_i[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_i[i] += 1
        
        B = [i for i in range(n) if n_i[i] == 0]
        k = 0
        while B:
            C = []
            for i in B:
                rank[i] = k
                for j in S[i]:
                    n_i[j] -= 1
                    if n_i[j] == 0:
                        C.append(j)
            B = C
            k += 1
        return rank

    def _crowding_distance(self, F, rank):
        n = len(F)
        dist = np.zeros(n)
        
        # Group by rank
        ranks = {}
        for i in range(n):
            r = rank[i]
            if r not in ranks:
                ranks[r] = []
            ranks[r].append(i)
        
        for r, indices in ranks.items():
            if len(indices) <= 2:
                for i in indices:
                    dist[i] = float('inf')
                continue
            
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_indices = sorted(indices, key=lambda i: F[i][m])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0]][m]
                f_max = F[sorted_indices[-1]][m]
                
                if f_max == f_min:
                    continue
                
                for k in range(1, len(sorted_indices) - 1):
                    i = sorted_indices[k]
                    dist[i] += (F[sorted_indices[k + 1]][m] - F[sorted_indices[k - 1]][m]) / (f_max - f_min)
        
        return dist

    def _select_next_gen(self, X, F, pop_size, rng):
        rank = self._non_dominated_sort(F)
        dist = self._crowding_distance(F, rank)
        
        # Sort by rank (asc) then distance (desc)
        order = np.lexsort((dist * -1, rank))
        
        # Select top pop_size
        selected_indices = order[:pop_size]
        
        return X[selected_indices], F[selected_indices]

    def _tournament_select(self, X, F, rng, k):
        rank = self._non_dominated_sort(F)
        dist = self._crowding_distance(F, rank)
        
        # Create a list of indices
        indices = list(range(len(X)))
        
        # Select k random candidates
        candidates = rng.choice(indices, k, replace=False)
        
        # Find the best candidate
        best_idx = candidates[0]
        for i in range(1, k):
            c = candidates[i]
            if rank[c] < rank[best_idx]:
                best_idx = c
            elif rank[c] == rank[best_idx]:
                if dist[c] > dist[best_idx]:
                    best_idx = c
        
        return X[best_idx]

    def _crossover(self, p1, p2, rng):
        # SBX Crossover
        child = p1.copy()
        for i in range(self.dim):
            if rng.random() < 0.5:
                child[i] = p2[i]
        return child

    def _mutate(self, x, lo, hi, rng):
        # Gaussian mutation
        sigma = 0.1
        x = x + sigma * rng.standard_normal(self.dim)
        x = np.clip(x, lo, hi)
        return x
