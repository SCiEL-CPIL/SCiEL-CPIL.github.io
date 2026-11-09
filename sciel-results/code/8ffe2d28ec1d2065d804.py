"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG9 with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # 280s safety margin within 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_dim = self.dim
        n_obj = self.n_obj
        
        # Population size: use a reasonable fraction of budget
        pop_size = min(100, max(20, self.budget // 2))
        
        # Initial random population
        X = lo + (hi - lo) * rng.random((pop_size, n_dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def non_dominated_sort(F):
            """Returns list of fronts (each front is a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated set
            nd = np.zeros(n, dtype=int)  # Dominance count
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1
            
            current_front = [i for i in range(n) if nd[i] == 0]
            rank = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                current_front = next_front
                rank += 1
            
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            m = n_obj
            for k in range(m):
                vals = F[indices, k]
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                fmin, fmax = vals[sorted_idx[0]], vals[sorted_idx[-1]]
                if fmax > fmin:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / (fmax - fmin)
            return cd

        # Main evolution loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Non-dominated sorting
            fronts = non_dominated_sort(F)
            
            # Select parents from first front(s)
            front0 = fronts[0]
            if len(front0) == 0:
                parent_idx = rng.integers(0, pop_size)
            elif len(front0) == 1:
                parent_idx = front0[0]
            else:
                # Tournament selection from first front
                i1, i2 = rng.integers(0, len(front0), size=2)
                # Use crowding distance for tie-breaking
                cd = crowding_distance(F, front0)
                parent_idx = front0[i1 if cd[i1] > cd[i2] else i2]
            
            # Create child with mutation
            child = X[parent_idx].copy()
            # Adaptive mutation rate
            progress = used / self.budget
            mut_rate = 0.5 * (1 - progress) + 0.1  # Decreases from 0.6 to 0.1
            for d in range(n_dim):
                if rng.random() < mut_rate:
                    child[d] += 0.1 * (hi[d] - lo[d]) * rng.standard_normal()
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst individual in last front
            last_front = fronts[-1]
            if len(last_front) > 1:
                # Replace individual with worst crowding distance in last front
                cd = crowding_distance(F, last_front)
                worst_local = np.argmin(cd)
                replace_idx = last_front[worst_local]
            else:
                replace_idx = last_front[0]
            
            X[replace_idx] = child
            F[replace_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
