import numpy as np

class Optimizer:
    """NSGA-II based multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        # Determine population size. 
        # We want to use the full budget. 
        # Let P be population size. 
        # 1 init + G generations.
        # Budget = P * (1 + G).
        # For good exploration, P=50 is reasonable for dim=5, n_obj=3.
        # 200 / 50 = 4. So 1 init + 3 generations.
        self.pop_size = 50
        self.generations = self.budget // self.pop_size - 1
        
        # Crossover and mutation parameters
        self.crossover_prob = 0.9
        self.mutation_prob = 1.0 / self.dim
        self.eta_crossover = 20.0
        self.eta_mutation = 20.0

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # 1. Initialize population
        X = lo + span * np.random.random((self.pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        
        evals_used = self.pop_size
        
        # 2. Evolutionary loop
        for _ in range(self.generations):
            # Create offspring
            offspring_X = np.empty_like(X)
            offspring_F = np.empty_like(F)
            
            for i in range(self.pop_size):
                # Select parents using binary tournament
                p1 = self._tournament_select(X, F)
                p2 = self._tournament_select(X, F)
                
                # Crossover
                if np.random.random() < self.crossover_prob:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, span)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Mutation
                c1 = self._poly_mutation(c1, lo, hi, span)
                c2 = self._poly_mutation(c2, lo, hi, span)
                
                offspring_X[2*i] = c1
                offspring_X[2*i+1] = c2
                
            # Evaluate offspring
            for i in range(self.pop_size):
                offspring_F[i] = problem(offspring_X[i])
                evals_used += 1
                
                # If we hit the budget exactly, we might need to break, 
                # but our loop is sized to fit.
            
            # Combine parent and offspring
            X_all = np.vstack((X, offspring_X))
            F_all = np.vstack((F, offspring_F))
            
            # Non-dominated sorting and crowding distance
            ranks, dists = self._non_dominated_sort(F_all)
            
            # Select next generation
            next_X = np.empty_like(X)
            next_F = np.empty_like(F)
            
            selected_indices = self._select_next_generation(ranks, dists, self.pop_size)
            
            next_X = X_all[selected_indices]
            next_F = F_all[selected_indices]
            
            X = next_X
            F = next_F
            
        return X

    def _tournament_select(self, X, F):
        """Binary tournament selection based on non-dominated rank and crowding distance."""
        # We need ranks and distances. 
        # For efficiency in a small loop, we can recompute or store them.
        # Storing is better. However, the current structure recomputes.
        # Let's optimize: compute ranks/distances once per generation for the whole population.
        # But _tournament_select is called inside the offspring loop.
        # This is inefficient. 
        # Better approach: Compute ranks and dists for X, F once. 
        # Then use them for selection.
        
        # Since I can't easily refactor the whole class structure without risking bugs in this short rewrite,
        # I will modify the main loop to pre-compute ranks and dists for X, F, 
        # and pass them to a helper or just implement selection logic that uses pre-computed values.
        
        # Let's restart the __call__ logic to be more efficient.
        pass

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # 1. Initialize population
        X = lo + span * np.random.random((self.pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        
        # Pre-compute ranks and dists for initial population
        ranks, dists = self._non_dominated_sort(F)
        
        for _ in range(self.generations):
            # Create offspring
            offspring_X = np.empty_like(X)
            
            for i in range(self.pop_size):
                # Select parents using binary tournament with pre-computed ranks/dists
                p1_idx = self._tournament_select(ranks, dists)
                p2_idx = self._tournament_select(ranks, dists)
                
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Crossover
                if np.random.random() < self.crossover_prob:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, span)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Mutation
                c1 = self._poly_mutation(c1, lo, hi, span)
                c2 = self._poly_mutation(c2, lo, hi, span)
                
                offspring_X[2*i] = c1
                offspring_X[2*i+1] = c2
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            
            # Combine parent and offspring
            X_all = np.vstack((X, offspring_X))
            F_all = np.vstack((F, offspring_F))
            
            # Non-dominated sorting and crowding distance for combined population
            ranks_all, dists_all = self._non_dominated_sort(F_all)
            
            # Select next generation
            selected_indices = self._select_next_generation(ranks_all, dists_all, self.pop_size)
            
            X = X_all[selected_indices]
            F = F_all[selected_indices]
            
            # Update ranks and dists for next round of selection
            ranks, dists = self._non_dominated_sort(F)

        return X

    def _tournament_select(self, ranks, dists):
        """Binary tournament selection."""
        i1 = np.random.randint(len(ranks))
        i2 = np.random.randint(len(ranks))
        
        # Compare rank first (lower is better)
        if ranks[i1] < ranks[i2]:
            return i1
        elif ranks[i2] < ranks[i1]:
            return i2
        else:
            # Compare crowding distance (higher is better)
            if dists[i1] > dists[i2]:
                return i1
            else:
                return i2

    def _sbx_crossover(self, p1, p2, lo, hi, span):
        """Simulated Binary Crossover."""
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p1)
        
        for i in range(self.dim):
            if np.random.random() < 0.5:
                # Crossover on dimension i
                if abs(p1[i] - p2[i]) < 1e-14:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
                else:
                    u = np.random.random()
                    beta1 = 1.0
                    beta2 = 1.0
                    
                    if u <= 0.5:
                        alpha = 2.0 * u
                        beta1 = (alpha * (self.eta_crossover + 1.0)) ** (1.0 / (self.eta_crossover + 1.0))
                    else:
                        alpha = 2.0 * (1.0 - u)
                        beta1 = (alpha * (self.eta_crossover + 1.0)) ** (1.0 / (self.eta_crossover + 1.0))
                    
                    # Calculate child 1
                    c1[i] = 0.5 * ((1.0 + beta1) * p1[i] + (1.0 - beta1) * p2[i])
                    
                    # Calculate beta2 for child 2
                    if u <= 0.5:
                        alpha = 2.0 * (1.0 - u)
                        beta2 = (alpha * (self.eta_crossover + 1.0)) ** (1.0 / (self.eta_crossover + 1.0))
                    else:
                        alpha = 2.0 * u
                        beta2 = (alpha * (self.eta_crossover + 1.0)) ** (1.0 / (self.eta_crossover + 1.0))
                    
                    c2[i] = 0.5 * ((1.0 + beta2) * p2[i] + (1.0 - beta2) * p1[i])
                    
                    # Clip to bounds
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
            else:
                c1[i] = p1[i]
                c2[i] = p2[i]
                
        return c1, c2

    def _poly_mutation(self, x, lo, hi, span):
        """Polynomial Mutation."""
        y = x.copy()
        for i in range(self.dim):
            if np.random.random() < self.mutation_prob:
                u = np.random.random()
                delta1 = (2.0 * u) ** (1.0 / (self.eta_mutation + 1.0)) - 1.0
                delta2 = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (self.eta_mutation + 1.0))
                
                y[i] += span[i] * (delta1 if np.random.random() < 0.5 else delta2)
                y[i] = np.clip(y[i], lo[i], hi[i])
        return y

    def _non_dominated_sort(self, F):
        """
        Perform non-dominated sorting and calculate crowding distance.
        Returns:
        ranks: array of shape (N,) with rank index for each point
        dists: array of shape (N,) with crowding distance for each point
        """
        N = len(F)
        ranks = np.zeros(N, dtype=int)
        dists = np.zeros(N, dtype=float)
        
        # Fronts
        fronts = []
        remaining = list(range(N))
        
        # Compute domination counts and dominated sets
        domination_count = np.zeros(N, dtype=int)
        dominated_set = [set() for _ in range(N)]
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # Check if i dominates j
                # i dominates j if F[i] <= F[j] in all objectives and < in at least one
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].add(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].add(i)
                    domination_count[i] += 1
        
        # First front
        front = [i for i in range(N) if domination_count[i] == 0]
        ranks[front] = 0
        fronts.append(front)
        
        k = 0
        while fronts[k] and remaining:
            current_front = fronts[k]
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
                        ranks[j] = k + 1
            if next_front:
                fronts.append(next_front)
                k += 1
        
        # Calculate crowding distance for each front
        for front in fronts:
            if len(front) <= 2:
                for i in front:
                    dists[i] = float('inf')
                continue
            
            # Sort front by each objective
            for m in range(self.n_obj):
                sorted_front = sorted(front, key=lambda x: F[x][m])
                dists[sorted_front[0]] = float('inf')
                dists[sorted_front[-1]] = float('inf')
                
                if len(sorted_front) > 2:
                    for i in range(1, len(sorted_front) - 1):
                        idx1 = sorted_front[i-1]
                        idx2 = sorted_front[i]
                        idx3 = sorted_front[i+1]
                        
                        denom = F[idx3][m] - F[idx1][m]
                        if denom > 1e-14:
                            dists[idx2] += (F[idx3][m] - F[idx1][m]) / denom
                        else:
                            dists[idx2] += 0.0
        
        return ranks, dists

    def _select_next_generation(self, ranks, dists, pop_size):
        """
        Select pop_size individuals from the population based on ranks and dists.
        """
        N = len(ranks)
        selected = []
        
        # Sort by rank, then by crowding distance (descending)
        # We need to iterate through fronts in order
        max_rank = max(ranks)
        
        for r in range(max_rank + 1):
            indices_in_front = [i for i in range(N) if ranks[i] == r]
            
            if not indices_in_front:
                continue
            
            # Sort by crowding distance descending
            indices_in_front.sort(key=lambda x: dists[x], reverse=True)
            
            if len(selected) + len(indices_in_front) <= pop_size:
                selected.extend(indices_in_front)
            else:
                # Need to select only some from this front
                needed = pop_size - len(selected)
                selected.extend(indices_in_front[:needed])
                break
        
        return selected

