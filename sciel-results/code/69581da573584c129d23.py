"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        N = 20  # Population size
        if self.budget < N:
            # Fallback if budget is very small
            X = lo + (hi - lo) * rng.random((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return X

        # 1. Initialization
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N

        # Helper: Fast Non-Dominated Sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            front = [0]
            fronts = [[]]
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                        n_dom[q] += 1
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        S[q].append(p)
                        n_dom[p] += 1
                
                if n_dom[p] == 0:
                    fronts[0].append(p)
                    front[0] += 1
            
            i = 0
            while front[i] > 0:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                if next_front:
                    fronts.append(next_front)
                    i += 1
                    front.append(len(next_front))
                else:
                    break
            return fronts

        # Helper: Crowding Distance
        def crowding_distance(F, front):
            if len(front) <= 2:
                return {i: float('inf') for i in front}
            m = F.shape[1]
            dist = {i: 0.0 for i in front}
            for k in range(m):
                sorted_front = sorted(front, key=lambda i: F[i, k])
                dist[sorted_front[0]] = float('inf')
                dist[sorted_front[-1]] = float('inf')
                f_min = F[sorted_front[0], k]
                f_max = F[sorted_front[-1], k]
                if f_max - f_min == 0:
                    continue
                for j in range(1, len(sorted_front) - 1):
                    idx = sorted_front[j]
                    dist[idx] += (F[sorted_front[j+1], k] - F[sorted_front[j-1], k]) / (f_max - f_min)
            return dist

        # Main Loop
        while used < self.budget:
            # Select Parents (Tournament)
            parents = []
            for _ in range(N):
                # Tournament selection size 2
                i, j = rng.integers(0, N, 2)
                # Compare dominance
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    parents.append(i)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    parents.append(j)
                else:
                    # Same rank? Just pick random for simplicity or use rank/crowding
                    # For efficiency in this small loop, random is okay if we rely on mutation
                    parents.append(i if rng.random() > 0.5 else j)
            
            # Crossover and Mutation
            offspring_X = []
            offspring_F = []
            
            # We need to generate N offspring to fill the next population
            # But we are limited by remaining budget.
            # Let's generate as many as we can, up to N.
            num_offspring = min(N, self.budget - used)
            
            for _ in range(num_offspawn):
                p1, p2 = rng.choice(parents, 2, replace=False)
                # Simulated Binary Crossover (SBX) simplified or just blend
                # Use simple linear blend crossover
                alpha = rng.random()
                child = alpha * X[p1] + (1 - alpha) * X[p2]
                
                # Gaussian Mutation
                # Sigma scales with bounds
                sigma = 0.2 * (hi - lo)
                child = child + sigma * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)
            
            if not offspring_X:
                break
                
            # Combine Parent and Offspring
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.vstack([F, np.array(offspring_F)])
            
            # Selection: Keep best N
            fronts = fast_non_dominated_sort(F_combined)
            
            X_new = []
            F_new = []
            
            for front in fronts:
                if len(X_new) + len(front) <= N:
                    # Add all
                    for idx in front:
                        X_new.append(X_combined[idx])
                        F_new.append(F_combined[idx])
                else:
                    # Need to trim this front
                    needed = N - len(X_new)
                    dists = crowding_distance(F_combined, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: dists[i], reverse=True)
                    for idx in sorted_front[:needed]:
                        X_new.append(X_combined[idx])
                        F_new.append(F_combined[idx])
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)

        return X
# EVOLVE-BLOCK-END
