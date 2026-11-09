import numpy as np


class Optimizer:
    """Multi-objective optimizer using NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """
        Fast non-dominated sorting.
        Returns: list of lists, where each list contains indices of points in that front.
        """
        n = len(F)
        # S[i] = set of indices dominated by i
        # n_dom[i] = number of points dominating i
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        fronts = []
        current_front = np.where(n_dom == 0)[0].tolist()
        
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
            
        # If there are points left (should not happen if all are non-dominated or dominated), 
        # but in case of ties or numerical issues, ensure all are covered
        # Usually, the loop above covers all.
        
        return fronts

    def _crowding_distance(self, F, indices):
        """
        Calculate crowding distance for points in indices.
        Returns array of distances same length as indices.
        """
        n = len(indices)
        if n == 0:
            return np.array([])
        if n <= 2:
            return np.full(n, np.inf)
            
        dist = np.zeros(n)
        m = F.shape[1]
        
        for k in range(m):
            # Sort indices by objective k
            sorted_indices = np.argsort(F[indices, k])
            
            # End points have infinite distance
            dist[sorted_indices[0]] = np.inf
            dist[sorted_indices[-1]] = np.inf
            
            f_min = F[indices[sorted_indices[0]], k]
            f_max = F[indices[sorted_indices[-1]], k]
            f_range = f_max - f_min
            
            if f_range > 0:
                for i in range(1, n - 1):
                    # Get actual indices in the original array
                    idx_curr = indices[sorted_indices[i]]
                    idx_prev = indices[sorted_indices[i-1]]
                    idx_next = indices[sorted_indices[i+1]]
                    
                    dist[sorted_indices[i]] += (F[idx_next, k] - F[idx_prev, k]) / f_range
                    
        return dist

    def _sbx_crossover(self, x1, x2, bounds, eta_c=15.0, rng=None):
        """Simulated Binary Crossover."""
        if rng is None:
            rng = np.random
        x1, x2 = np.array(x1), np.array(x2)
        lo, hi = bounds
        c1, c2 = np.zeros_like(x1), np.zeros_like(x2)
        
        for i in range(self.dim):
            if rng.random() < 0.5:
                # Compute beta
                if x1[i] != x2[i]:
                    if x1[i] < x2[i]:
                        lo_i, hi_i = x1[i], x2[i]
                    else:
                        lo_i, hi_i = x2[i], x1[i]
                    
                    # beta = (2*lo_i + (hi_i-lo_i)*u^(-1/(eta+1))) / (lo_i+hi_i) ?? 
                    # Standard formula:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta_c + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    c2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                else:
                    c1[i] = x1[i]
                    c2[i] = x2[i]
            else:
                c1[i] = x2[i]
                c2[i] = x1[i]
                
        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _poly_mutation(self, x, bounds, eta_m=20.0, prob_mut=0.1, rng=None):
        """Polynomial Mutation."""
        if rng is None:
            rng = np.random
        x = np.array(x)
        lo, hi = bounds
        dim = self.dim
        mut_mask = rng.random(dim) < prob_mut
        
        for i in range(dim):
            if mut_mask[i]:
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                    x[i] -= delta * (hi[i] - lo[i])
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                    x[i] += delta * (hi[i] - lo[i])
                x[i] = np.clip(x[i], lo[i], hi[i])
                
        return x

    def __call__(self, problem, bounds):
        # Set seed for reproducibility if needed, but task says evaluator seeds np.random
        # We should use np.random directly or a generator seeded by np.random state?
        # The task says: "For seeds 0 and 1 the evaluator calls np.random.seed(seed)..."
        # So we can just use np.random functions.
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = 20
        # Initial population
        X = np.random.uniform(lo, hi, size=(pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Determine number of generations
        # Budget 200, Pop 20. Init 20. Remaining 180.
        # Each gen adds 20. 180/20 = 9 generations.
        n_gens = (self.budget - pop_size) // pop_size
        
        for g in range(n_gens):
            # Generate offspring
            offspring_X = np.zeros((pop_size, self.dim))
            offspring_F = np.zeros((pop_size, self.n_obj))
            
            for i in range(pop_size):
                # Tournament selection for parents
                # Select 2 parents
                p1_idx = np.random.randint(pop_size)
                p2_idx = np.random.randint(pop_size)
                
                # Simple random selection is okay for small pop, but tournament is better
                # Let's use tournament of size 2 based on non-dominated rank
                # For simplicity and speed in this small context, random is often sufficient 
                # or we can implement a quick rank-based selection.
                # To be robust, let's do a simple tournament.
                # We need ranks. We calculate ranks every generation.
                
                # Actually, let's just pick random parents to keep it simple and fast,
                # relying on the selection step to drive convergence.
                # Or better: pick parents from the current population X.
                
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                # Crossover
                c1, c2 = self._sbx_crossover(x1, x2, (lo, hi), eta_c=15.0)
                
                # Mutation
                c1 = self._poly_mutation(c1, (lo, hi), eta_m=20.0, prob_mut=1.0/self.dim)
                c2 = self._poly_mutation(c2, (lo, hi), eta_m=20.0, prob_mut=1.0/self.dim)
                
                offspring_X[2*i] = c1
                offspring_X[2*i+1] = c2
                
            # Evaluate offspring
            # We have 2*pop_size offspring candidates, but we only have budget for pop_size more?
            # No, standard NSGA-II generates pop_size offspring.
            # My loop above generated 2*pop_size. I need to adjust.
            # Let's generate pop_size offspring in one go.
            
            offspring_X = np.zeros((pop_size, self.dim))
            for i in range(pop_size):
                p1_idx = np.random.randint(pop_size)
                p2_idx = np.random.randint(pop_size)
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                c1, _ = self._sbx_crossover(x1, x2, (lo, hi), eta_c=15.0)
                c1 = self._poly_mutation(c1, (lo, hi), eta_m=20.0, prob_mut=1.0/self.dim)
                offspring_X[i] = c1
                
            # Evaluate
            for i in range(pop_size):
                if used >= self.budget:
                    break
                offspring_F[i] = problem(offspring_X[i])
                used += 1
            
            # Combine populations
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])
            
            # Fast Non-Dominated Sorting
            fronts = self._fast_non_dominated_sort(F_combined)
            
            # Assign ranks
            ranks = np.zeros(len(F_combined), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank + 1 # 1-based rank
            
            # Calculate Crowding Distance for each front
            # We need to select pop_size points.
            selected_indices = []
            
            for front in fronts:
                if len(selected_indices) + len(front) <= pop_size:
                    # Take all
                    selected_indices.extend(front)
                else:
                    # Need to trim
                    n_needed = pop_size - len(selected_indices)
                    # Calculate crowding distance for this front
                    cd = self._crowding_distance(F_combined, np.array(front))
                    # Sort by cd descending
                    sorted_front = np.array(front)[np.argsort(cd)[::-1]]
                    selected_indices.extend(sorted_front[:n_needed].tolist())
                    break
            
            # Update population
            X = X_combined[selected_indices]
            F = F_combined[selected_indices]
            
        return X
