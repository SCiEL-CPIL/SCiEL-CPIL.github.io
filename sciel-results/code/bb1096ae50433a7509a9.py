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
        deadline = start_time + 300.0  # 300s limit, keep margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        pop_size = 40
        if self.budget < pop_size * 2:
             pop_size = max(10, self.budget // 4)

        # Initialize population
        # Use np.random for consistency with external seed setting
        X = lo + (hi - lo) * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II parameters
        eta_c = 15.0  # Distribution index for crossover
        eta_m = 20.0  # Distribution index for mutation
        p_c = 0.9     # Crossover probability

        while used + pop_size <= self.budget:
            # Check time guard
            if time.time() > deadline - 5.0:
                break

            # 1. Compute ranks and crowding for current X, F
            ranks, crowding = self._non_dominated_sort(F)
            
            # 2. Select parents using tournament selection based on ranks and crowding
            parents_idx = np.zeros(pop_size, dtype=int)
            for i in range(pop_size):
                idx1 = np.random.randint(pop_size)
                idx2 = np.random.randint(pop_size)
                # Compare: lower rank is better. If same rank, higher crowding is better.
                if ranks[idx1] < ranks[idx2]:
                    parents_idx[i] = idx1
                elif ranks[idx2] < ranks[idx1]:
                    parents_idx[i] = idx2
                else:
                    if crowding[idx1] >= crowding[idx2]:
                        parents_idx[i] = idx1
                    else:
                        parents_idx[i] = idx2
            
            # 3. Generate offspring
            offspring = np.zeros((pop_size, self.dim))
            for i in range(0, pop_size, 2):
                if np.random.rand() < p_c:
                    p1 = X[parents_idx[i]]
                    p2 = X[parents_idx[i+1]]
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c)
                    offspring[i] = c1
                    offspring[i+1] = c2
                else:
                    offspring[i] = X[parents_idx[i]].copy()
                    offspring[i+1] = X[parents_idx[i+1]].copy()
            
            # 4. Mutation
            for i in range(pop_size):
                offspring[i] = self._polynomial_mutation(offspring[i], lo, hi, eta_m)
            
            # 5. Evaluate offspring
            F_offspring = np.array([problem(x) for x in offspring])
            used += pop_size
            
            # 6. Combine parents and offspring
            X_combined = np.vstack([X, offspring])
            F_combined = np.vstack([F, F_offspring])
            
            # 7. Non-dominated sort and selection
            ranks_comb, crowding_comb = self._non_dominated_sort(F_combined)
            
            # Select best pop_size individuals
            # Sort by rank, then by crowding (descending)
            # We need to pick the first pop_size from a list sorted by (rank asc, crowding desc)
            # Create indices sorted by rank
            sorted_indices = np.lexsort(((-1 * crowding_comb), ranks_comb))
            
            X = X_combined[sorted_indices[:pop_size]]
            F = F_combined[sorted_indices[:pop_size]]

        return X

    def _non_dominated_sort(self, F):
        """
        Compute non-dominated ranks and crowding distances.
        Returns:
            ranks: array of ranks (0 is best)
            crowding: array of crowding distances
        """
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n)
        
        # Non-dominated sorting
        S = [[] for _ in range(n)] # S[i] = set of individuals dominated by i
        n_dominates = np.zeros(n, dtype=int) # n_dominates[i] = number of individuals dominating i
        front = []
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                # Minimization: i dominates j if F[i] <= F[j] and F[i] != F[j]
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominates[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominates[i] += 1
        
        # Find first front
        for i in range(n):
            if n_dominates[i] == 0:
                ranks[i] = 0
                front.append(i)
        
        k = 0
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            k += 1
            front = next_front
        
        # Calculate crowding distance for each front
        # Group indices by rank
        max_rank = max(ranks) if n > 0 else 0
        fronts = [[] for _ in range(max_rank + 1)]
        for i in range(n):
            fronts[ranks[i]].append(i)
        
        for front_indices in fronts:
            if len(front_indices) <= 2:
                # If front has 1 or 2 individuals, crowding is infinity (or large)
                for i in front_indices:
                    crowding[i] = np.inf
                continue
            
            m = self.n_obj
            # Sort front by each objective
            for k_obj in range(m):
                # Sort indices in front_indices by F[obj]
                sorted_idx = sorted(front_indices, key=lambda x: F[x, k_obj])
                # Assign inf to first and last
                crowding[sorted_idx[0]] = np.inf
                crowding[sorted_idx[-1]] = np.inf
                # Calculate range
                f_min = F[sorted_idx[0], k_obj]
                f_max = F[sorted_idx[-1], k_obj]
                if f_max - f_min == 0:
                    continue
                for i in range(1, len(sorted_idx) - 1):
                    idx = sorted_idx[i]
                    idx_prev = sorted_idx[i-1]
                    idx_next = sorted_idx[i+1]
                    # Add to existing crowding distance
                    # Note: standard NSGA-II adds to the distance. 
                    # We initialize to 0, so we add.
                    # But wait, we set first/last to inf. 
                    # For middle points, we add (F[next] - F[prev]) / (f_max - f_min)
                    crowding[idx] += (F[idx_next, k_obj] - F[idx_prev, k_obj]) / (f_max - f_min)
        
        return ranks, crowding

    def _sbx_crossover(self, p1, p2, lo, hi, eta):
        c1 = p1.copy()
        c2 = p2.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                # Calculate beta
                if abs(p2[i] - p1[i]) < 1e-12:
                    beta = 1.0
                else:
                    r = np.random.rand()
                    if r <= 0.5:
                        alpha = 2.0 * r
                        beta = (alpha) ** (1.0 / (eta + 1.0))
                    else:
                        alpha = 2.0 * (1.0 - r)
                        beta = (alpha) ** (1.0 / (eta + 1.0))
                
                # c1 = 0.5 * ((1+beta)*p1 + (1-beta)*p2)
                # c2 = 0.5 * ((1-beta)*p1 + (1+beta)*p2)
                c1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
                c2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
        
        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, eta):
        x_mut = x.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.5: # Mutation probability per gene is usually 1/dim, but 0.5 is safe? 
                # Standard NSGA-II uses p_m = 1/dim. Let's use that.
                pass # We'll implement standard p_m = 1/dim below
        
        # Reset to standard implementation
        p_m = 1.0 / self.dim
        for i in range(self.dim):
            if np.random.rand() < p_m:
                # Calculate delta
                if x[i] == lo[i]:
                    y_low = 0.0
                    y_high = (hi[i] - lo[i]) / (hi[i] - lo[i]) # Normalized? 
                    # Standard formula:
                    # if x == lo, y_low = 0, y_high = (hi - lo) / (hi - lo) = 1? 
                    # Actually, the polynomial mutation formula works on the normalized space [0,1] or directly?
                    # Let's stick to the direct formula from Deb's book.
                    pass
        
        # Let's implement a simpler, robust mutation: Gaussian or Uniform within bounds?
        # Polynomial is better. Let's implement it correctly.
        
        x_mut = x.copy()
        for i in range(self.dim):
            if np.random.rand() < p_m:
                # Normalize x to [0,1]
                range_i = hi[i] - lo[i]
                if range_i == 0:
                    continue
                x_norm = (x[i] - lo[i]) / range_i
                
                y_low = 0.0
                y_high = 1.0
                delta_1 = (y_high - y_low)
                
                if x_norm == y_low:
                    y_low = 0.0
                    delta_2 = y_high - y_low
                elif x_norm == y_high:
                    y_high = 1.0
                    delta_2 = y_high - y_low
                else:
                    delta_2 = y_high - y_low
                
                r = np.random.rand()
                if r < 0.5:
                    xy = 2.0 * r
                    c = (xy) ** (1.0 / (eta + 1.0))
                    delta = c * delta_2
                    y_new = x_norm - delta
                else:
                    xy = 2.0 * (1.0 - r)
                    c = (xy) ** (1.0 / (eta + 1.0))
                    delta = c * delta_2
                    y_new = x_norm + delta
                
                # Clip y_new to [0,1]
                y_new = np.clip(y_new, 0.0, 1.0)
                x_mut[i] = lo[i] + y_new * range_i
        
        return x_mut
# EVOLVE-BLOCK-END
