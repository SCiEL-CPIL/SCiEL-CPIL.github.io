"""Multi-objective NSGA-II optimizer for WFG7 with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II with SBX crossover and polynomial mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        """Return True if a dominates b."""
        return np.all(a <= b) and np.any(a < b)

    def _nsga2_rank(self, F):
        """
        Fast Non-dominated Sorting.
        Returns ranks (array of shape (N,)) and cdists (array of shape (N,)).
        """
        N = len(F)
        ranks = np.zeros(N, dtype=int)
        cdists = np.zeros(N)
        
        # Domination info
        domination_count = np.zeros(N, dtype=int)
        dominated_set = [[] for _ in range(N)]
        
        # Calculate domination
        for i in range(N):
            for j in range(i + 1, N):
                if self._dominates(F[i], F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(F[j], F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
        
        # Front 0
        front = [i for i in range(N) if domination_count[i] == 0]
        front_idx = 0
        ranks[front] = 0
        
        while front:
            next_front = []
            for i in front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        ranks[j] = front_idx + 1
                        next_front.append(j)
            front_idx += 1
            front = next_front
        
        # Calculate Crowding Distance for each front
        max_rank = int(np.max(ranks)) if N > 0 else -1
        for r in range(max_rank + 1):
            front_indices = np.where(ranks == r)[0]
            n_front = len(front_indices)
            
            if n_front <= 2:
                cdists[front_indices] = np.inf
                continue
                
            # For each objective, sort and calculate distances
            for k in range(self.n_obj):
                obj_vals = F[front_indices, k]
                sorted_local_idx = np.argsort(obj_vals)
                sorted_global_idx = front_indices[sorted_local_idx]
                sorted_obj_vals = obj_vals[sorted_local_idx]
                
                # Assign infinity to extremes
                cdists[sorted_global_idx[0]] = np.inf
                cdists[sorted_global_idx[-1]] = np.inf
                
                # Normalize objective range for this front
                range_val = sorted_obj_vals[-1] - sorted_obj_vals[0]
                if range_val < 1e-12:
                    continue
                
                # Calculate distances for intermediate points
                for i in range(1, n_front - 1):
                    dist = (sorted_obj_vals[i+1] - sorted_obj_vals[i-1]) / range_val
                    cdists[sorted_global_idx[i]] += dist
        
        return ranks, cdists

    def _tournament(self, ranks, cdists):
        """Binary tournament selection."""
        N = len(ranks)
        i = np.random.randint(N)
        j = np.random.randint(N)
        
        if ranks[i] < ranks[j]:
            return i
        elif ranks[j] < ranks[i]:
            return j
        else:
            if cdists[i] > cdists[j]:
                return i
            elif cdists[j] > cdists[i]:
                return j
            else:
                return i

    def _sbx_crossover(self, p1, p2, lo, hi, eta):
        """Simulated Binary Crossover."""
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p2)
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    u1 = (p1[i] - lo[i]) / (hi[i] - lo[i])
                    u2 = (p2[i] - lo[i]) / (hi[i] - lo[i])
                    
                    if u1 <= u2:
                        if np.random.rand() < 0.5:
                            beta = (u1 / u2) ** (eta + 1) if u2 > 0 else 1.0
                            c1_norm = 0.5 * (u1 + u2 - beta * (u2 - u1))
                            c2_norm = 0.5 * (u1 + u2 + beta * (u2 - u1))
                        else:
                            beta = (u2 / u1) ** (eta + 1) if u1 > 0 else 1.0
                            c1_norm = 0.5 * (u1 + u2 + beta * (u2 - u1))
                            c2_norm = 0.5 * (u1 + u2 - beta * (u2 - u1))
                    else:
                        if np.random.rand() < 0.5:
                            beta = (u2 / u1) ** (eta + 1) if u1 > 0 else 1.0
                            c1_norm = 0.5 * (u1 + u2 - beta * (u2 - u1))
                            c2_norm = 0.5 * (u1 + u2 + beta * (u2 - u1))
                        else:
                            beta = (u1 / u2) ** (eta + 1) if u2 > 0 else 1.0
                            c1_norm = 0.5 * (u1 + u2 + beta * (u2 - u1))
                            c2_norm = 0.5 * (u1 + u2 - beta * (u2 - u1))
                    
                    c1[i] = lo[i] + c1_norm * (hi[i] - lo[i])
                    c2[i] = lo[i] + c2_norm * (hi[i] - lo[i])
                else:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                c1[i] = p1[i]
                c2[i] = p2[i]
        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        
        return c1, c2

    def _poly_mutation(self, x, lo, hi, eta, p_m):
        """Polynomial Mutation."""
        x_mut = x.copy()
        for i in range(self.dim):
            if np.random.rand() < p_m:
                u = np.random.rand()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                
                x_norm = (x[i] - lo[i]) / (hi[i] - lo[i])
                x_norm += delta * x_norm * (1 - x_norm)
                x_mut[i] = lo[i] + x_norm * (hi[i] - lo[i])
        
        x_mut = np.clip(x_mut, lo, hi)
        return x_mut

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # With budget 200 and dim 12, N=20 allows for ~10 generations (20 evals per gen)
        # This is a reasonable balance for exploration/exploitation
        N = 20
        
        # 1. Initialize population
        X = lo + (hi - lo) * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Parameters for SBX and Polynomial Mutation
        eta_c = 20.0
        eta_m = 20.0
        p_m = 1.0 / self.dim
        
        while evals + N <= self.budget:
            if time.time() > deadline:
                break
                
            # 2. Generate Offspring
            offspring_X = np.zeros_like(X)
            
            for i in range(0, N, 2):
                # Select parents using binary tournament selection based on NSGA-II ranking
                ranks, cdists = self._nsga2_rank(F)
                
                idx1 = self._tournament(ranks, cdists)
                idx2 = self._tournament(ranks, cdists)
                
                p1 = X[idx1]
                p2 = X[idx2]
                
                # SBX Crossover
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c)
                
                # Polynomial Mutation
                c1 = self._poly_mutation(c1, lo, hi, eta_m, p_m)
                c2 = self._poly_mutation(c2, lo, hi, eta_m, p_m)
                
                offspring_X[i] = c1
                offspring_X[i+1] = c2
            
            # Evaluate Offspring
            F_off = np.array([problem(x) for x in offspring_X])
            evals += N
            
            # 3. Combine and Select
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, F_off])
            
            # Rank combined population
            ranks, cdists = self._nsga2_rank(F_combined)
            
            # Select next N
            # Sort by rank, then by crowding distance (descending)
            # np.lexsort sorts by the last key first.
            # We want primary key: rank (ascending), secondary key: -cdists (ascending, which is cdists descending)
            sort_idx = np.lexsort((-cdists, ranks))
            
            next_X = X_combined[sort_idx[:N]]
            next_F = F_combined[sort_idx[:N]]
            
            X = next_X
            F = next_F
            
        return X
# EVOLVE-BLOCK-END
