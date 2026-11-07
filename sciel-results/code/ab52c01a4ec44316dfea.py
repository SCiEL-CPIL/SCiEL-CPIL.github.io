import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Differential Evolution (DE) with 
    Non-Dominated Sorting selection (MOEA/DE)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Population size
        pop_size = 40
        if self.budget < pop_size:
            pop_size = max(2, self.budget)

        # Initialize population using random sampling
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F_mat):
            n = len(F_mat)
            # Vectorized dominance: F[i] dominates F[j] if all(F[i] <= F[j]) and any(F[i] < F[j])
            # Build dominance matrix
            # A[i,j] = True if i dominates j
            le = F_mat[:, np.newaxis, :] <= F_mat[np.newaxis, :, :]  # (n, n, m)
            lt = F_mat[:, np.newaxis, :] < F_mat[np.newaxis, :, :]   # (n, n, m)
            dom_mask = np.all(le, axis=2) & np.any(lt, axis=2)       # (n, n)
            np.fill_diagonal(dom_mask, False)
            
            S = [[] for _ in range(n)]
            n_dominating = np.zeros(n, dtype=int)
            for i in range(n):
                S[i] = np.where(dom_mask[i])[0].tolist()
                n_dominating[i] = np.sum(dom_mask[:, i])
            
            fronts = []
            front0 = [i for i in range(n) if n_dominating[i] == 0]
            fronts.append(front0)
            while front0:
                next_front = []
                for i in front0:
                    for j in S[i]:
                        n_dominating[j] -= 1
                        if n_dominating[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front0 = next_front
            return fronts

        def crowding_distance(F_mat, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                vals = F_mat[front_indices, m]
                min_v, max_v = vals.min(), vals.max()
                if max_v - min_v < 1e-15:
                    continue
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / (max_v - min_v)
            return cd

        def select_next_population(F_all, X_all, pop_size):
            """Select next population using NSGA-II logic."""
            fronts = fast_non_dominated_sort(F_all)
            new_indices = []
            for front in fronts:
                if len(new_indices) + len(front) <= pop_size:
                    new_indices.extend(front)
                else:
                    remaining = pop_size - len(new_indices)
                    if remaining <= 0:
                        break
                    cd = crowding_distance(F_all, front)
                    sorted_front = [front[j] for j in np.argsort(-cd)]
                    new_indices.extend(sorted_front[:remaining])
                    break
            
            while len(new_indices) < pop_size:
                new_indices.append(rng.integers(len(F_all)))
            
            return X_all[new_indices], F_all[new_indices]

        # DE Parameters - adaptive
        F_scale_start = 0.9
        F_scale_end = 0.5
        CR = 0.5

        n_generations = (self.budget - used) // pop_size
        gen = 0

        while used < self.budget:
            if time.time() > deadline:
                break

            n_children = min(pop_size, self.budget - used)
            if n_children <= 0:
                break

            # Adaptive F: linear decay from F_scale_start to F_scale_end
            if n_generations > 0:
                F_scale = F_scale_start - (F_scale_start - F_scale_end) * (gen / max(1, n_generations - 1))
            else:
                F_scale = F_scale_end

            X_c = np.zeros((n_children, self.dim))
            
            for i in range(n_children):
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                
                v = X[r1] + F_scale * (X[r2] - X[r3])
                v = np.clip(v, lo, hi)
                
                j_rand = rng.integers(self.dim)
                trial = X[i].copy()
                
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    
                X_c[i] = trial

            F_c = np.array([problem(x) for x in X_c])
            used += n_children

            X_all = np.vstack([X, X_c])
            F_all = np.vstack([F, F_c])
            
            X, F = select_next_population(F_all, X_all, pop_size)
            gen += 1

        return X
