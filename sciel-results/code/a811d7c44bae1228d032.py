"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _fast_nondominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        dom_cnt = [0] * n
        fronts = [[]]
        
        for p in range(n):
            for q in range(p + 1, n):
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                    dom_cnt[q] += 1
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    S[q].append(p)
                    dom_cnt[p] += 1
            if dom_cnt[p] == 0:
                fronts[0].append(p)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    dom_cnt[q] -= 1
                    if dom_cnt[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return fronts[:-1]

    def _crowding_distance(self, F, indices):
        if len(indices) == 0:
            return np.zeros(0)
        cd = np.zeros(len(indices))
        n_obj = self.n_obj
        for m in range(n_obj):
            vals = F[indices, m]
            min_val = np.min(vals)
            max_val = np.max(vals)
            if max_val == min_val:
                cd[:] = np.inf
                continue
            idx_min = np.argmin(vals)
            idx_max = np.argmax(vals)
            cd[idx_min] = np.inf
            cd[idx_max] = np.inf
            sorted_indices = np.argsort(vals)
            for k in range(1, len(indices) - 1):
                cd[sorted_indices[k]] += (vals[sorted_indices[k+1]] - vals[sorted_indices[k-1]]) / (max_val - min_val)
        return cd

    def _tournament(self, ranks, cd, k=2):
        best = 0
        for _ in range(k):
            idx = np.random.randint(0, len(ranks))
            if ranks[idx] < ranks[best]:
                best = idx
            elif ranks[idx] == ranks[best] and cd[idx] > cd[best]:
                best = idx
        return best

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        
        pop_size = 20
        if self.budget < pop_size:
            pop_size = self.budget
        
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        while used < self.budget:
            if time.time() > deadline - 5.0:
                break
                
            fronts = self._fast_nondominated_sort(F)
            ranks = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
            
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cd[front] = self._crowding_distance(F, front)
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            while len(offspring_X) < pop_size:
                p1_idx = self._tournament(ranks, cd)
                p2_idx = self._tournament(ranks, cd)
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # SBX crossover
                if rng.random() < 0.9:
                    alpha = 20
                    beta1 = rng.random()
                    if beta1 <= 0.5:
                        beta = (2 * beta1) ** (1.0 / (alpha + 1))
                    else:
                        beta = (2 * (1 - beta1)) ** (1.0 / (alpha + 1))
                    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
                else:
                    c1 = p1.copy()
                    c2 = p2.copy()
                
                # Mutation
                eta = 20
                for i in range(self.dim):
                    if rng.random() < 1.0 / self.dim:
                        delta1 = (2 * rng.random()) ** (1.0 / (eta + 1)) - 1
                        c1[i] += delta1 * (hi[i] - lo[i])
                for i in range(self.dim):
                    if rng.random() < 1.0 / self.dim:
                        delta2 = (2 * rng.random()) ** (1.0 / (eta + 1)) - 1
                        c2[i] += delta2 * (hi[i] - lo[i])
                
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                
                fc1 = problem(c1)
                fc2 = problem(c2)
                used += 2
                if used > self.budget:
                    # If we exceeded budget, we might have to discard the second one if we can't fit it,
                    # but the contract says "More than 200 evaluations... makes the whole result invalid".
                    # So we must stop BEFORE exceeding.
                    # Let's adjust the loop condition to be safer.
                    # Actually, the loop `while used < self.budget` is checked at the start of the outer loop.
                    # Inside, we add 2. If used was budget-1, we add 2 -> budget+1. Invalid.
                    # We need to ensure we don't exceed.
                    break
                
                offspring_X.append(c1)
                offspring_X.append(c2)
                offspring_F.append(fc1)
                offspring_F.append(fc2)
            
            if len(offspring_X) == 0:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])
            
            fronts_combined = self._fast_nondominated_sort(F_combined)
            ranks_combined = np.zeros(len(F_combined), dtype=int)
            for i, front in enumerate(fronts_combined):
                for idx in front:
                    ranks_combined[idx] = i
            
            cd_combined = np.zeros(len(F_combined))
            for front in fronts_combined:
                if len(front) > 0:
                    cd_combined[front] = self._crowding_distance(F_combined, front)
            
            # Select best pop_size
            new_X = []
            new_F = []
            for front in fronts_combined:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                else:
                    # Need to pick from this front
                    remaining = pop_size - len(new_X)
                    front_cd = cd_combined[front]
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: front_cd[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
        
        return X
# EVOLVE-BLOCK-END
