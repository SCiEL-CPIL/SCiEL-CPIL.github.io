"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II with Differential Evolution mutation for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size: 20 is a good balance for 200 budget
        N = 20
        
        # Initialize population
        rng = np.random.default_rng(0)
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        evals = N
        
        # DE Parameters
        F_mut = 0.8
        CR = 0.9
        
        # Main loop
        while evals < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            X_off = np.empty_like(X)
            F_off = np.empty_like(F)
            
            for i in range(N):
                if evals >= self.budget or time.time() > deadline:
                    break
                
                # Select 3 distinct parents for DE/rand/1
                idxs = rng.choice(N, size=3, replace=False)
                p1, p2, p3 = X[idxs[0]], X[idxs[1]], X[idxs[2]]
                
                # Mutation
                v = p1 + F_mut * (p2 - p3)
                
                # Crossover
                j_rand = rng.integers(self.dim)
                mask = rng.random(self.dim) < CR
                mask[j_rand] = True
                child = np.where(mask, v, X[i])
                
                # Clip to bounds
                child = np.clip(child, lo, hi)
                
                # Evaluate
                fc = problem(child)
                evals += 1
                
                X_off[i] = child
                F_off[i] = fc
            
            # If we didn't generate a full offspring set, break
            if evals < self.budget:
                # Combine and select
                X_combined = np.vstack([X, X_off])
                F_combined = np.vstack([F, F_off])
                
                # Non-dominated sorting
                ranks, fronts = self._non_dominated_sort(F_combined)
                
                # Crowding distance
                cd = self._crowding_distance(F_combined, ranks)
                
                # Select next N parents
                X_next = np.empty_like(X)
                F_next = np.empty_like(F)
                
                count = 0
                for front in fronts:
                    if count + len(front) <= N:
                        # Take all
                        for idx in front:
                            X_next[count] = X_combined[idx]
                            F_next[count] = F_combined[idx]
                            count += 1
                    else:
                        # Take only what fits, sorted by crowding distance
                        remaining = N - count
                        # Sort front by crowding distance (descending)
                        front_indices = sorted(front, key=lambda x: cd[x], reverse=True)
                        for idx in front_indices[:remaining]:
                            X_next[count] = X_combined[idx]
                            F_next[count] = F_combined[idx]
                            count += 1
                        break
                
                X = X_next
                F = F_next
            
            if evals >= self.budget:
                break

        return X

    def _non_dominated_sort(self, F):
        """Perform non-dominated sorting."""
        n = len(F)
        S = [[] for _ in range(n)]  # Domination sets
        n_dominates = np.zeros(n, dtype=int)  # Number of solutions dominated
        rank = np.zeros(n, dtype=int)
        
        fronts = []
        i = 0
        while i < n:
            front = []
            for j in range(n):
                if rank[j] == 0 and n_dominates[j] == 0:
                    front.append(j)
                    rank[j] = 1
                    i += 1
            
            if not front:
                break
                
            fronts.append(front)
            
            for p in front:
                for q in range(n):
                    if self._dominates(F[p], F[q]):
                        n_dominates[q] -= 1
                        if n_dominates[q] == 0:
                            rank[q] = 1
                            i += 1
                            
        return rank, fronts

    def _dominates(self, f1, f2):
        """Check if f1 dominates f2 (minimization)."""
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _crowding_distance(self, F, rank):
        """Calculate crowding distance for each solution."""
        n = len(F)
        cd = np.zeros(n)
        
        # Group by rank
        ranks = np.unique(rank)
        for r in ranks:
            indices = np.where(rank == r)[0]
            if len(indices) <= 2:
                cd[indices] = np.inf
                continue
                
            for k in range(self.n_obj):
                # Sort by objective k
                sorted_indices = indices[np.argsort(F[indices, k])]
                cd[sorted_indices[0]] = np.inf
                cd[sorted_indices[-1]] = np.inf
                
                f_min = F[sorted_indices[0], k]
                f_max = F[sorted_indices[-1], k]
                range_k = f_max - f_min
                
                if range_k < 1e-10:
                    continue
                    
                for i in range(1, len(sorted_indices) - 1):
                    prev_idx = sorted_indices[i-1]
                    next_idx = sorted_indices[i+1]
                    cd[sorted_indices[i]] += (F[next_idx, k] - F[prev_idx, k]) / range_k
                    
        return cd
# EVOLVE-BLOCK-END
