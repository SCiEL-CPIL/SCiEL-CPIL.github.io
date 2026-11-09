"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective optimizer with non-dominated sorting and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        pop_size = min(50, self.budget // 4)
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            rank = [0] * n
            front = [0]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    rank[i] = 0
                    front[0] += 1
            
            k = 0
            current_front = list(range(front[0]))
            while len(current_front) > 0:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                current_front = next_front
            return rank

        def crowding_distance(F, rank, indices):
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            cd = [0.0] * n
            for m in range(self.n_obj):
                vals = [F[idx][m] for idx in indices]
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                min_v = vals[sorted_idx[0]]
                max_v = vals[sorted_idx[-1]]
                if max_v - min_v < 1e-12:
                    continue
                for i in range(1, n - 1):
                    pos = sorted_idx[i]
                    prev_pos = sorted_idx[i - 1]
                    next_pos = sorted_idx[i + 1]
                    cd[pos] += (vals[next_pos] - vals[prev_pos]) / (max_v - min_v)
            return cd

        while used < self.budget:
            if time.time() > deadline:
                break
                
            rank = non_dominated_sort(F)
            
            # Select parents based on rank and crowding distance
            fronts = {}
            for i in range(len(rank)):
                r = rank[i]
                if r not in fronts:
                    fronts[r] = []
                fronts[r].append(i)
            
            # Sort fronts by rank
            sorted_fronts = sorted(fronts.keys())
            
            # Build selection pool with crowding
            selection_pool = []
            for r in sorted_fronts:
                indices = fronts[r]
                cd = crowding_distance(F, r, indices)
                for i, idx in enumerate(indices):
                    selection_pool.append((r, -cd[i], idx))
            
            # Sort by rank then by crowding (higher crowding first)
            selection_pool.sort(key=lambda x: (x[0], -x[1]))
            
            # DE-based offspring generation
            n_children = min(pop_size, self.budget - used)
            for _ in range(n_children):
                if used >= self.budget or time.time() > deadline:
                    break
                    
                # Select 3 distinct parents
                p1 = selection_pool[rng.integers(len(selection_pool))][2]
                p2 = selection_pool[rng.integers(len(selection_pool))][2]
                p3 = selection_pool[rng.integers(len(selection_pool))][2]
                
                F_cr = 0.5 + 0.5 * rng.random()
                CR = 0.5
                j_rand = rng.integers(self.dim)
                
                child = np.zeros(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        child[j] = X[p1][j] + F_cr * (X[p2][j] - X[p3][j])
                    else:
                        child[j] = X[p1][j]
                
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                
                # Replace worst individual in the population
                # Use a simple strategy: replace a dominated individual if possible, else random
                dominated_mask = np.array([np.any(np.all(F <= fc, axis=1) & np.any(F < fc, axis=1)) for _ in range(len(F))])
                if np.any(dominated_mask):
                    candidates = np.where(dominated_mask)[0]
                    worst_idx = candidates[rng.integers(len(candidates))]
                else:
                    worst_idx = rng.integers(len(F))
                
                X[worst_idx] = child
                F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
