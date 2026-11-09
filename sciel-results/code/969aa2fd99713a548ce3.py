"""Rewritten optimizer for ZDT1 using a decomposition-based approach (NSGA-II style with crowding)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n = self.budget

        # Population size: use a reasonable fraction of budget
        pop_size = max(20, min(100, n // 4))
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def fast_non_dominated_sort(F):
            """Perform fast non-dominated sort, return list of fronts."""
            m = len(F)
            S = [set() for _ in range(m)]
            n_dom = [0] * m
            rank = [0] * m
            fronts = []
            
            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    rank[i] = 0
                    if not fronts or len(fronts[-1]) == 0:
                        fronts.append([i])
                    else:
                        # Check if last front is empty (shouldn't happen)
                        if not fronts:
                            fronts.append([i])
                        else:
                            fronts[-1].append(i)
            
            # Fix the front assignment - redo more carefully
            fronts = []
            current_front = [i for i in range(m) if rank[i] == 0]
            if current_front:
                fronts.append(current_front)
            
            k = 0
            while k < len(fronts):
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            
            return fronts

        def crowding_distance(F, front_indices):
            """Calculate crowding distance for individuals in a front."""
            if len(front_indices) <= 2:
                return np.full(len(front_indices), np.inf)
            
            m = len(front_indices)
            dist = np.zeros(m)
            n_obj = F.shape[1]
            
            for k in range(n_obj):
                f_vals = F[front_indices, k]
                sorted_idx = np.argsort(f_vals)
                dist_sorted = np.zeros(m)
                dist_sorted[0] = np.inf
                dist_sorted[-1] = np.inf
                f_range = f_vals[sorted_idx[-1]] - f_vals[sorted_idx[0]]
                if f_range > 1e-15:
                    for i in range(1, m - 1):
                        dist_sorted[i] = (f_vals[sorted_idx[i + 1]] - f_vals[sorted_idx[i - 1]]) / f_range
                # Map back to original indices
                for i, idx in enumerate(sorted_idx):
                    pos_in_front = front_indices.index(idx)
                    dist[pos_in_front] += dist_sorted[i]
            
            return dist

        # Main loop
        while used < n:
            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(F)
            
            # Create next generation
            next_X = []
            next_F = []
            
            # Add individuals from fronts until we have pop_size
            for front in fronts:
                if len(next_F) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X[idx])
                        next_F.append(F[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(next_F)
                    cd = crowding_distance(F, front)
                    # Sort by crowding distance descending
                    sorted_order = np.argsort(-cd)
                    for i in range(remaining):
                        idx = front[sorted_order[i]]
                        next_X.append(X[idx])
                        next_F.append(F[idx])
                    break
            
            # If we still need more (shouldn't happen), fill with random
            while len(next_F) < pop_size:
                x = lo + (hi - lo) * rng.random(self.dim)
                fx = problem(x)
                next_X.append(x)
                next_F.append(fx)
                used += 1
            
            X = np.array(next_X)
            F = np.array(next_F)
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size and used < n:
                # Select parents using tournament selection based on non-domination rank
                front_ranks = {}
                for fi, front in enumerate(fronts):
                    for idx in front:
                        front_ranks[idx] = fi
                
                p1_idx = self._tournament_select(front_ranks, F, rng)
                p2_idx = self._tournament_select(front_ranks, F, rng)
                
                # SBX crossover
                if rng.random() < 0.9:
                    c1, c2 = self._sbx(X[p1_idx], X[p2_idx], lo, hi, rng)
                else:
                    c1 = X[p1_idx].copy()
                    c2 = X[p2_idx].copy()
                
                # Mutation
                for c in [c1, c2]:
                    if rng.random() < 0.9:
                        self._polynomial_mutation(c, lo, hi, rng)
                    c = np.clip(c, lo, hi)
                
                f1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                if used >= n:
                    break
                
                f2 = problem(c2)
                used += 1
                offspring_X.append(c2)
                offspring_F.append(f2)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)]) if offspring_X else X
            combined_F = np.vstack([F, np.array(offspring_F)]) if offspring_F else F
            
            # Fast non-dominated sort on combined
            combined_fronts = fast_non_dominated_sort(combined_F)
            
            # Select next generation
            next_X = []
            next_F = []
            for front in combined_fronts:
                if len(next_F) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    remaining = pop_size - len(next_F)
                    cd = crowding_distance(combined_F, front)
                    sorted_order = np.argsort(-cd)
                    for i in range(remaining):
                        idx = front[sorted_order[i]]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
        
        return X

    def _tournament_select(self, front_ranks, F, rng, k=2):
        """Tournament selection based on non-domination rank."""
        candidates = list(front_ranks.keys())
        selected = []
        for _ in range(k):
            idx = candidates[rng.integers(len(candidates))]
            selected.append(idx)
        
        # Pick the one with best (lowest) front rank
        best = min(selected, key=lambda i: front_ranks[i])
        return best

    def _sbx(self, x1, x2, lo, hi, rng, eta=20.0):
        """Simulated Binary Crossover."""
        c1 = x1.copy()
        c2 = x2.copy()
        
        for i in range(len(x1)):
            if rng.random() < 0.5:
                if abs(x2[i] - x1[i]) > 1e-15:
                    if rng.random() <= 1.0 / (eta + 1.0):
                        beta = (2.0 * rng.random()) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2.0 * rng.random())) ** (1.0 / (eta + 1.0))
                    c1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                    c2[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, rng, eta=20.0, prob=1.0 / 30.0):
        """Polynomial mutation in place."""
        for i in range(len(x)):
            if rng.random() < prob:
                dx = x[i] - lo[i]
                dxh = hi[i] - x[i]
                rand = rng.random()
                if rand < 0.5:
                    xy = (2.0 * rand) ** (1.0 / (eta + 1.0))
                    delta = (1.0 - xy) ** (eta + 1.0)
                    x[i] -= delta * dx
                else:
                    xy = (2.0 * (1.0 - rand)) ** (1.0 / (eta + 1.0))
                    delta = (1.0 - xy) ** (eta + 1.0)
                    x[i] += delta * dxh
# EVOLVE-BLOCK-END
