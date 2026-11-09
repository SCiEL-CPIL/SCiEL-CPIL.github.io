"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective search with non-dominated sorting and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    @staticmethod
    def _dominance_sort(F):
        """Return non-dominated sort fronts as list of lists of indices."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        n_dominated = np.zeros(n, dtype=int)
        fronts = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominated[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dominated[i] += 1
        front0 = [i for i in range(n) if n_dominated[i] == 0]
        fronts.append(front0)
        while front0:
            next_front = []
            for i in front0:
                for j in S[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front0 = next_front
        return fronts

    @staticmethod
    def _crowding_distance(F, front_indices):
        """Compute crowding distance for indices in front."""
        m = len(front_indices)
        if m <= 2:
            return np.full(m, np.inf)
        dist = np.zeros(m)
        for k in range(F.shape[1]):
            vals = F[front_indices, k]
            v_min, v_max = vals.min(), vals.max()
            if v_max == v_min:
                continue
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for idx in range(1, m - 1):
                dist[order[idx]] += (vals[order[idx + 1]] - vals[order[idx - 1]]) / (v_max - v_min)
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget

        # Population size: balance exploration and exploitation
        pop_size = max(10, min(40, budget // 5))
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        FCR = 0.8  # Differential evolution crossover rate
        CR = 0.9   # Crossover probability for DE
        K = 0.5    # DE scale factor

        while used < budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sorting
            fronts = self._dominance_sort(F)
            
            # Select parents from best front(s)
            front0 = fronts[0]
            if len(front0) < 2:
                front0 = list(range(pop_size))
            
            # Generate offspring using DE
            parent_indices = rng.choice(front0, size=min(3, len(front0)), replace=False)
            p1, p2, p3 = parent_indices
            
            # DE mutation
            v = X[p1] + K * (X[p2] - X[p3])
            
            # DE crossover
            jrand = rng.integers(dim)
            trial = np.empty(dim)
            for j in range(dim):
                if rng.random() < CR or j == jrand:
                    trial[j] = v[j]
                else:
                    trial[j] = X[p1][j]
            
            trial = np.clip(trial, lo, hi)
            
            # Evaluate
            ft = problem(trial)
            used += 1
            
            # Tournament selection: keep if it dominates or has better crowding
            # Simple approach: replace worst in population by sum of objectives
            # But better: use non-dominated sorting to select replacement
            if np.any(np.all(F <= ft, axis=1) & np.any(F < ft, axis=1)):
                # ft dominates at least one, find the worst dominated
                dominated_by_ft = [i for i in range(pop_size) 
                                   if np.all(ft <= F[i]) and np.any(ft < F[i])]
                if dominated_by_ft:
                    # Replace the one with highest sum
                    sums = np.sum(F[dominated_by_ft], axis=1)
                    worst_idx = dominated_by_ft[np.argmax(sums)]
                    X[worst_idx] = trial
                    F[worst_idx] = ft
                else:
                    # Shouldn't happen, but fallback
                    worst = int(np.argmax(np.sum(F, axis=1)))
                    X[worst] = trial
                    F[worst] = ft
            else:
                # Check if ft is non-dominated
                dominates_any = any(np.all(ft <= F[i]) and np.any(ft < F[i]) for i in range(pop_size))
                if dominates_any:
                    # Replace the dominated one with worst sum
                    dominated_by_ft = [i for i in range(pop_size) 
                                       if np.all(ft <= F[i]) and np.any(ft < F[i])]
                    sums = np.sum(F[dominated_by_ft], axis=1)
                    worst_idx = dominated_by_ft[np.argmax(sums)]
                    X[worst_idx] = trial
                    F[worst_idx] = ft
                else:
                    # ft is non-dominated and doesn't dominate anyone
                    # Replace the point with worst crowding distance in worst front
                    # Approximate: replace point with highest sum that is dominated
                    dominated = [i for i in range(pop_size) 
                                 if np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1))]
                    if dominated:
                        sums = np.sum(F[dominated], axis=1)
                        worst_idx = dominated[np.argmax(sums)]
                    else:
                        worst_idx = int(np.argmax(np.sum(F, axis=1)))
                    X[worst_idx] = trial
                    F[worst_idx] = ft

        return X
# EVOLVE-BLOCK-END
