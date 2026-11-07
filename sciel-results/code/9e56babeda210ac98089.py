"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II based multi-objective optimizer with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        pop_size = 24
        if self.budget < pop_size:
            pop_size = self.budget
            
        # 1. Initial Population using Latin Hypercube Sampling
        # We generate pop_size points
        X = np.zeros((pop_size, self.dim))
        for i in range(pop_size):
            # For each dimension, pick a unique stratified point
            # This is a simple LHS approximation
            pass
            
        # Simpler LHS: 
        # Shuffle permutations for each dimension
        X = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(pop_size)
            # Map indices 0..pop_size-1 to intervals [i/pop_size, (i+1)/pop_size)
            # Add random offset within interval
            offsets = np.random.random(pop_size)
            vals = (perm + offsets) / pop_size
            X[:, d] = lo[d] + (hi[d] - lo[d]) * vals
            
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Check budget
        if used >= self.budget:
            return X

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # 2. Non-dominated Sorting
            # Returns ranks (lower is better) and crowding distances
            ranks, crowding = self.fast_non_dominated_sort(F)
            
            # 3. Selection: Tournament Selection based on rank and crowding
            # We select parents for crossover
            parents = []
            # We want to generate pop_size offspring
            # So we need pop_size parents (or 2*pop_size if binary tournament picks 2)
            # Let's do binary tournament to pick 2 parents per offspring
            offspring_X = []
            offspring_F = []
            
            # Calculate how many generations we can afford
            # Each iteration costs pop_size evaluations
            # remaining = self.budget - used
            # if remaining < pop_size: break (or do partial)
            
            if self.budget - used < pop_size:
                # Do a few single evaluations if budget allows? 
                # Or just stop. NSGA-II works in generations.
                # Let's just stop to keep it clean, or do random mutations if 1 left.
                # Given 200 budget and 24 pop, we have ~8 generations.
                break

            # Generate offspring
            for _ in range(pop_size):
                p1 = self.tournament_select(ranks, crowding)
                p2 = self.tournament_select(ranks, crowding)
                
                # SBX Crossover
                c1, c2 = self.sbx_crossover(X[p1], X[p2], lo, hi)
                
                # Mutation
                c1 = self.polynomial_mutation(c1, lo, hi)
                c2 = self.polynomial_mutation(c2, lo, hi)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
                used += 2
                if used > self.budget:
                    break
                if time.time() > deadline:
                    break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X[:pop_size]) # Ensure size
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += len(offspring_X)
            
            # 4. Combine Parent and Offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])
            
            # 5. Selection for next generation
            # Non-dominated sort on combined population
            ranks_all, crowding_all = self.fast_non_dominated_sort(F_all)
            
            # Select next pop_size individuals
            # Sort by rank, then by crowding distance (descending)
            # We need to pick pop_size from 2*pop_size
            # Create indices sorted by (rank, -crowding)
            sort_indices = np.lexsort((-crowding_all, ranks_all))
            
            # We need to be careful: if the k-th rank has more individuals than slots left,
            # we only take some of them.
            X_next = np.zeros((pop_size, self.dim))
            F_next = np.zeros((pop_size, self.n_obj))
            
            for i in range(pop_size):
                idx = sort_indices[i]
                X_next[i] = X_all[idx]
                F_next[i] = F_all[idx]
                
            X = X_next
            F = F_next

        return X

    def fast_non_dominated_sort(self, F):
        """
        Returns ranks (array of ints, 0 is best) and crowding distances.
        """
        n = len(F)
        if n == 0:
            return np.array([]), np.array([])
            
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n)
        
        # Dominance relationships
        S = [[] for _ in range(n)] # S[i] contains j dominated by i
        n_dominators = np.zeros(n, dtype=int)
        
        # Calculate dominance
        for i in range(n):
            for j in range(i+1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominators[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominators[i] += 1
                    
        # First front
        front = [i for i in range(n) if n_dominators[i] == 0]
        if not front:
             # All dominated? This shouldn't happen for min problems unless equal.
             # If all equal, all are non-dominated.
             if np.all(np.all(F == F[0], axis=1)):
                 front = list(range(n))
             else:
                 # Fallback
                 front = [0]
        
        k = 0
        while front:
            next_front = []
            for i in front:
                ranks[i] = k
                for j in S[i]:
                    n_dominators[j] -= 1
                    if n_dominators[j] == 0:
                        next_front.append(j)
            k += 1
            front = next_front
        
        # Calculate Crowding Distance for each front
        # Group indices by rank
        fronts = {}
        for i in range(n):
            if ranks[i] not in fronts:
                fronts[ranks[i]] = []
            fronts[ranks[i]].append(i)
            
        for rank, indices in fronts.items():
            if len(indices) <= 2:
                for i in indices:
                    crowding[i] = float('inf')
                continue
                
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_indices = sorted(indices, key=lambda i: F[i, m])
                crowding[sorted_indices[0]] = float('inf')
                crowding[sorted_indices[-1]] = float('inf')
                
                range_val = F[sorted_indices[-1], m] - F[sorted_indices[0], m]
                if range_val == 0:
                    continue
                    
                for i in range(1, len(sorted_indices) - 1):
                    idx_curr = sorted_indices[i]
                    idx_prev = sorted_indices[i-1]
                    idx_next = sorted_indices[i+1]
                    crowding[idx_curr] += (F[idx_next, m] - F[idx_prev, m]) / range_val
                    
        return ranks, crowding

    def tournament_select(self, ranks, crowding, k=2):
        """Binary tournament selection."""
        indices = np.random.randint(0, len(ranks), k)
        best_idx = -1
        best_rank = float('inf')
        best_crowding = -1
        
        for idx in indices:
            if ranks[idx] < best_rank:
                best_rank = ranks[idx]
                best_crowding = crowding[idx]
                best_idx = idx
            elif ranks[idx] == best_rank:
                if crowding[idx] > best_crowding:
                    best_crowding = crowding[idx]
                    best_idx = idx
                    
        return best_idx

    def sbx_crossover(self, x1, x2, lo, hi, prob=0.9, eta=20.0):
        """Simulated Binary Crossover."""
        c1 = x1.copy()
        c2 = x2.copy()
        
        if np.random.rand() < prob:
            for i in range(self.dim):
                if x1[i] != x2[i]:
                    if x1[i] < x2[i]:
                        y1, y2 = x1[i], x2[i]
                    else:
                        y1, y2 = x2[i], x1[i]
                    
                    range_val = hi[i] - lo[i]
                    if range_val == 0:
                        continue
                        
                    # Calculate beta
                    if np.random.rand() < 0.5:
                        beta = (2.0 * y1 / range_val) ** (1.0 / (eta + 1.0)) if y1 > lo[i] else 1.0
                        # Actually standard SBX formulas:
                        # beta1 = (2*y1)^(1/(eta+1)) if y1 < 0.5*(lo+hi) ... wait, usually normalized.
                        # Let's use standard implementation from NSGA-II paper
                        # y1, y2 are the parents.
                        # beta = (2 * min(y1, y2) / (range))^(1/(eta+1)) ? No.
                        
                        # Correct SBX:
                        # If y1 < y2:
                        #   beta1 = (2*y1 / (y1+y2))^(1/(eta+1)) if y1 < 0.5*(y1+y2) ... 
                        #   Actually simpler:
                        #   beta = (2 * min(y1, y2) / (max(y1, y2) - min(y1, y2) + range)) ... no.
                        
                        # Let's use the standard formula:
                        # beta1 = (2 * y1)^(1/(eta+1)) if y1 < 0.5 else (2 * (1-y1))^(1/(eta+1))
                        # This assumes y is in [0,1].
                        
                        # Let's normalize to [0,1]
                        y1_n = (y1 - lo[i]) / range_val
                        y2_n = (y2 - lo[i]) / range_val
                        
                        if np.random.rand() < 0.5:
                            if y1_n < 0.5:
                                beta = (2.0 * y1_n) ** (1.0 / (eta + 1.0))
                            else:
                                beta = (2.0 * (1.0 - y1_n)) ** (1.0 / (eta + 1.0))
                            c1[i] = 0.5 * ((1.0 + beta) * y1 + (1.0 - beta) * y2)
                            c2[i] = 0.5 * ((1.0 + 1.0/beta) * y2 + (1.0 - 1.0/beta) * y1)
                        else:
                            if y2_n < 0.5:
                                beta = (2.0 * y2_n) ** (1.0 / (eta + 1.0))
                            else:
                                beta = (2.0 * (1.0 - y2_n)) ** (1.0 / (eta + 1.0))
                            c2[i] = 0.5 * ((1.0 + beta) * y2 + (1.0 - beta) * y1)
                            c1[i] = 0.5 * ((1.0 + 1.0/beta) * y1 + (1.0 - 1.0/beta) * y2)
                            
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def polynomial_mutation(self, x, lo, hi, prob=0.1, eta=20.0):
        """Polynomial Mutation."""
        x_mut = x.copy()
        for i in range(self.dim):
            if np.random.rand() < prob:
                range_val = hi[i] - lo[i]
                if range_val == 0:
                    continue
                y = x_mut[i]
                y_n = (y - lo[i]) / range_val
                delta1 = (y_n) ** (eta + 1.0)
                delta2 = (1.0 - y_n) ** (eta + 1.0)
                r = np.random.rand()
                if r < 0.5:
                    xy = (2.0 * r + (1.0 - 2.0 * r) * (1.0 - delta1)) ** (1.0 / (eta + 1.0))
                    deltaq = (xy - 1.0) * range_val
                else:
                    xy = (2.0 * (1.0 - r) + (2.0 * r - 1.0) * (1.0 - delta2)) ** (1.0 / (eta + 1.0))
                    deltaq = (xy - 1.0) * range_val
                x_mut[i] += deltaq
        return np.clip(x_mut, lo, hi)

# EVOLVE-BLOCK-END
