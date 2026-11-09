import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        N = 20
        
        # Initialize
        X = lo + (hi - lo) * np.random.random((N, dim))
        # Ensure initial points are finite
        X = np.nan_to_num(X, nan=(lo + hi) / 2.0, posinf=hi, neginf=lo)
        F = np.array([problem(x) for x in X])
        eval_count = N
        
        # NSGA-II Loop
        while eval_count + N <= self.budget:
            # Check deadline
            if time.time() > deadline - 5:
                break
            
            # Generate offspring
            offspring_X = np.zeros_like(X)
            for i in range(0, N, 2):
                idx1 = self._tournament(F, 2)
                idx2 = self._tournament(F, 2)
                c1, c2 = self._sbx_crossover_pair(X[idx1], X[idx2], lo, hi)
                c1 = self._poly_mutation(c1, lo, hi)
                c2 = self._poly_mutation(c2, lo, hi)
                offspring_X[i] = c1
                offspring_X[i+1] = c2
            
            # Evaluate offspring
            # Ensure offspring are finite
            for i in range(N):
                if not np.all(np.isfinite(offspring_X[i])):
                    offspring_X[i] = (lo + hi) / 2.0
            offspring_F = np.array([problem(x) for x in offspring_X])
            eval_count += N
            
            # Combine
            pop_X = np.vstack([X, offspring_X])
            pop_F = np.vstack([F, offspring_F])
            
            # Select next population
            selected_indices = self._nsga2_select(pop_F, N)
            X = pop_X[selected_indices]
            F = pop_F[selected_indices]

        return X

    def _tournament(self, F, k=2):
        n = len(F)
        candidates = np.random.choice(n, k, replace=False)
        best = candidates[0]
        for c in candidates[1:]:
            if np.all(F[best] <= F[c]) and np.any(F[best] < F[c]):
                continue
            elif np.all(F[c] <= F[best]) and np.any(F[c] < F[best]):
                best = c
            else:
                # Non-dominated, keep current best
                pass
        return best

    def _sbx_crossover_pair(self, x1, x2, lo, hi):
        eta = 20
        c1 = np.zeros_like(x1)
        c2 = np.zeros_like(x2)
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    val1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    val2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                    c1[i] = np.clip(val1, lo[i], hi[i])
                    c2[i] = np.clip(val2, lo[i], hi[i])
                else:
                    c1[i] = x1[i]
                    c2[i] = x2[i]
            else:
                c1[i] = x1[i]
                c2[i] = x2[i]
        return c1, c2

    def _poly_mutation(self, x, lo, hi):
        eta = 20
        pm = 1.0 / self.dim
        x = x.copy()
        for i in range(self.dim):
            if np.random.rand() < pm:
                delta1 = (x[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - x[i]) / (hi[i] - lo[i])
                u = np.random.rand()
                if u < 0.5:
                    base = 1 + 2 * (u - delta1)
                    if base < 0:
                        base = 0.0
                    xy = base ** (1 / (eta + 1))
                    deltaq = xy - 1
                else:
                    base = 1 + 2 * (delta2 - u)
                    if base < 0:
                        base = 0.0
                    xy = base ** (1 / (eta + 1))
                    deltaq = 1 - xy
                x[i] += deltaq * (hi[i] - lo[i])
                x[i] = np.clip(x[i], lo[i], hi[i])
                if not np.isfinite(x[i]):
                    x[i] = (lo[i] + hi[i]) / 2.0
        return x

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        ranks = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
        
        front = 0
        current_front = [i for i in range(n) if domination_count[i] == 0]
        for i in current_front:
            ranks[i] = 0
        
        while current_front:
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        ranks[j] = front + 1
                        next_front.append(j)
            front += 1
            current_front = next_front
            
        return ranks

    def _crowding_distance(self, F, indices):
        if len(indices) <= 2:
            return np.full(len(indices), np.inf)
        
        n = len(indices)
        dist = np.zeros(n)
        n_obj = self.n_obj
        
        for k in range(n_obj):
            sorted_indices = np.argsort(F[indices, k])
            sorted_vals = F[indices[sorted_indices], k]
            
            dist[sorted_indices[0]] = np.inf
            dist[sorted_indices[-1]] = np.inf
            
            if sorted_vals[-1] > sorted_vals[0]:
                for i in range(1, n - 1):
                    dist[sorted_indices[i]] += (sorted_vals[i+1] - sorted_vals[i-1]) / (sorted_vals[-1] - sorted_vals[0])
        
        return dist

    def _nsga2_select(self, F, N):
        n = len(F)
        ranks = self._fast_non_dominated_sort(F)
        
        selected = []
        front_idx = 0
        while front_idx < N:
            front_indices = np.where(ranks == front_idx)[0]
            if len(front_indices) + len(selected) <= N:
                selected.extend(front_indices)
            else:
                num_to_select = N - len(selected)
                cd = self._crowding_distance(F, front_indices)
                sorted_cd = np.argsort(cd)[::-1]
                selected.extend(front_indices[sorted_cd[:num_to_select]])
                break
            front_idx += 1
            
        return np.array(selected)
