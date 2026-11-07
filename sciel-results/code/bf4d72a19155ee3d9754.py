import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        # Set population size to a reasonable fraction of the budget
        # For budget=200, dim=4, n_obj=2, a pop size of 20 allows 10 generations
        self.pop_size = 20

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Random number generator
        rng = np.random.default_rng(42)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((self.pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        
        used = self.pop_size
        N = self.pop_size
        
        while used < self.budget:
            # Combine parents and offspring
            # For simplicity in this loop structure, we generate N offspring from N parents
            # Then select the best N from the 2N combined
            # However, standard NSGA-II generates N offspring, combines with N parents (2N total), selects N.
            
            # 1. Non-dominated sorting of current population X
            ranks, crowding = self._nondominated_sort(F)
            
            # 2. Generate offspring
            offspring_X = np.empty_like(X)
            for i in range(N):
                # Binary tournament selection
                p1 = self._tournament_select(ranks, crowding, rng)
                p2 = self._tournament_select(ranks, crowding, rng)
                
                # SBX Crossover
                c1, c2 = self._sbx(X[p1], X[p2], lo, hi, rng)
                
                # Polynomial Mutation
                c1 = self._polynomial_mutation(c1, lo, hi, rng)
                c2 = self._polynomial_mutation(c2, lo, hi, rng)
                
                offspring_X[2*i] = c1
                offspring_X[2*i+1] = c2
            
            # Evaluate offspring
            # We need to be careful with the budget.
            # We have N offspring to evaluate.
            # If used + N > budget, we can only evaluate (budget - used) offspring.
            # But standard NSGA-II replaces the whole population.
            # Let's stick to full generations if possible.
            # Budget is 200, Pop is 20. 10 full generations fit exactly.
            
            if used + N > self.budget:
                # Handle the tail end if budget doesn't divide evenly
                # For this specific task (200 budget, 20 pop), it divides evenly.
                # But for robustness:
                n_eval = self.budget - used
                offspring_X = offspring_X[:n_eval]
            else:
                n_eval = N
            
            F_off = np.array([problem(x) for x in offspring_X])
            used += n_eval
            
            # Combine parents and offspring
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, F_off])
            
            # 3. Non-dominated sorting of combined population
            ranks_comb, crowding_comb = self._nondominated_sort(F_comb)
            
            # 4. Select best N individuals for next generation
            # Sort by rank, then by crowding distance (descending)
            # We need to pick N individuals.
            indices = np.argsort(ranks_comb)
            # Stable sort by rank
            sorted_indices = np.argsort(ranks_comb, kind='mergesort')
            
            next_X = []
            next_F = []
            for idx in sorted_indices:
                if len(next_X) >= N:
                    break
                # Check if we should add this individual
                # In standard NSGA-II, we add entire fronts until we exceed N, then fill the last front with best crowding.
                # The simple argsort by rank then crowding works if we sort by (rank, -crowding)
                # Let's create a composite key
                pass
            
            # More efficient selection:
            # Sort by rank (asc), then by crowding (desc)
            # We can't easily do multi-key sort with numpy argsort directly without a struct or complex trick.
            # Let's use a list of tuples.
            sort_keys = [(ranks_comb[i], -crowding_comb[i]) for i in range(len(X_comb))]
            sorted_indices = np.argsort(sort_keys)
            
            selected_indices = sorted_indices[:N]
            X = X_comb[selected_indices]
            F = F_comb[selected_indices]

        return X

    def _nondominated_sort(self, F):
        """
        Perform non-dominated sorting and calculate crowding distance.
        Returns:
        ranks: array of rank for each point (0 is best)
        crowding: array of crowding distance for each point
        """
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n)
        
        S = [[] for _ in range(n)]  # S[i] contains indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # n_dom[i] is number of points dominating i
        
        # Determine dominance relations
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
        
        # First front
        front = [i for i in range(n) if n_dom[i] == 0]
        k = 0
        while front:
            next_front = []
            for i in front:
                ranks[i] = k
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            front = next_front
            k += 1
        
        # Calculate crowding distance for each front
        for k in range(len(ranks)):
            # Get indices in this front
            front_indices = np.where(ranks == k)[0]
            if len(front_indices) <= 2:
                # If only 1 or 2 points, crowding is infinity
                for i in front_indices:
                    crowding[i] = np.inf
                continue
            
            for m in range(self.n_obj):
                # Sort front by m-th objective
                sorted_front = front_indices[np.argsort(F[front_indices, m])]
                # Assign infinite crowding to boundary points
                crowding[sorted_front[0]] = np.inf
                crowding[sorted_front[-1]] = np.inf
                
                # Normalize objective range
                f_min = F[sorted_front[0], m]
                f_max = F[sorted_front[-1], m]
                if f_max - f_min == 0:
                    continue
                
                for i in range(1, len(sorted_front) - 1):
                    idx_prev = sorted_front[i-1]
                    idx_next = sorted_front[i+1]
                    idx_curr = sorted_front[i]
                    delta = (F[idx_next, m] - F[idx_prev, m]) / (f_max - f_min)
                    crowding[idx_curr] += delta
        
        return ranks, crowding

    def _tournament_select(self, ranks, crowding, rng):
        """Binary tournament selection."""
        i = rng.integers(len(ranks))
        j = rng.integers(len(ranks))
        
        if ranks[i] < ranks[j]:
            return i
        elif ranks[j] < ranks[i]:
            return j
        else:
            if crowding[i] > crowding[j]:
                return i
            else:
                return j

    def _sbx(self, p1, p2, lo, hi, rng, eta=15.0, prob=0.9):
        """Simulated Binary Crossover."""
        c1 = p1.copy()
        c2 = p2.copy()
        
        if rng.random() > prob:
            return c1, c2
            
        for i in range(self.dim):
            if rng.random() < 0.5:
                # Calculate beta
                if abs(p1[i] - p2[i]) < 1e-14:
                    beta = 1.0
                else:
                    r = rng.random()
                    if r <= 0.5:
                        beta = (2 * r) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1.0 - r))) ** (1.0 / (eta + 1.0))
                    
                    c1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
                    c2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
                    
                    # Clip to bounds
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, rng, eta=20.0):
        """Polynomial Mutation."""
        x_mut = x.copy()
        for i in range(self.dim):
            if rng.random() < 1.0 / self.dim:
                r = rng.random()
                y = x_mut[i]
                yl = lo[i]
                yh = hi[i]
                
                if r < 0.5:
                    xy = (y - yl) / (yh - yl)
                    mut_pow = 2.0 + (1.0 / (xy + 1.0) * eta) if xy > 0 else 2.0 + eta
                    u = 1.0 - (1.0 - r) * 2.0
                    delta1 = (2.0 * u) ** (1.0 / mut_pow) - 1.0
                    x_mut[i] = y + delta1 * (yh - yl)
                else:
                    xy = (yh - y) / (yh - yl)
                    mut_pow = 2.0 + (1.0 / (xy + 1.0) * eta) if xy > 0 else 2.0 + eta
                    u = 1.0 - r * 2.0
                    delta2 = 1.0 - (2.0 * u) ** (1.0 / mut_pow)
                    x_mut[i] = y + delta2 * (yh - yl)
                
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut
