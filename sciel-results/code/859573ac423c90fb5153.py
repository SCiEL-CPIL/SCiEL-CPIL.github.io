import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # Budget is 200. Let's use a population size that allows several generations.
        # N = 20, Generations = (200 - 20) / 20 = 9. Total 10 steps.
        # N = 10, Generations = (200 - 10) / 10 = 19. Total 20 steps.
        # Smaller N allows more generations, potentially better convergence, but worse diversity.
        # Let's try N=20.
        N = 20
        if N > self.budget:
            N = self.budget
            
        # Initialize population
        X = np.random.uniform(lo, hi, (N, self.dim))
        F = np.array([problem(x) for x in X])
        used_evals = N
        
        # NSGA-II parameters
        # Crossover probability
        pc = 0.9
        # Crossover distribution index (eta_c)
        eta_c = 15.0
        # Mutation probability
        pm = 1.0 / self.dim
        # Mutation distribution index (eta_m)
        eta_m = 20.0
        
        while used_evals < self.budget:
            # Generate offspring population
            X_off = np.zeros((N, self.dim))
            F_off = np.zeros((N, self.n_obj))
            
            for i in range(N):
                # Select parents using tournament selection based on non-dominated sorting
                # We need to sort the current population first
                # To save time, we can approximate or just use random selection if budget is very tight,
                # but NSGA-II requires sorting. Let's implement a lightweight sort.
                
                # Actually, sorting every generation for small N is cheap.
                # Let's do non-dominated sorting on current X, F
                ranks, crowding = self._nsga2_sort(X, F, N)
                
                # Tournament selection for parent 1
                p1_idx = self._tournament_select(ranks, crowding, N, 2)
                # Tournament selection for parent 2
                p2_idx = self._tournament_select(ranks, crowding, N, 2)
                
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                # Crossover
                if np.random.rand() < pc:
                    c1, c2 = self._sbx_crossover(x1, x2, lo, hi, eta_c)
                else:
                    c1, c2 = x1.copy(), x2.copy()
                
                # Mutation
                c1 = self._poly_mutation(c1, lo, hi, pm, eta_m)
                
                # Evaluate offspring
                f1 = problem(c1)
                used_evals += 1
                if used_evals > self.budget:
                    break
                    
                X_off[i] = c1
                F_off[i] = f1
                
                # If we have budget for the second offspring, we could generate it,
                # but standard NSGA-II generates N offspring from N pairs.
                # To keep it simple and within budget, we generate one offspring per parent pair slot.
                # The loop runs N times, generating N offspring.
                
            # Combine parent and offspring populations
            X_all = np.vstack((X, X_off))
            F_all = np.vstack((F, F_off))
            
            # Non-dominated sort the combined population
            ranks_all, crowding_all = self._nsga2_sort(X_all, F_all, 2 * N)
            
            # Select next generation
            # We want to select N individuals from the 2N
            # Sort by rank, then by crowding distance
            # Create an index array sorted by rank, then crowding
            sort_idx = np.lexsort((crowding_all, ranks_all))
            # We want the first N in this sorted order (lowest rank, highest crowding)
            # Note: lexsort sorts by last key first. So (crowding, ranks) means primary ranks, secondary crowding.
            # We want lowest ranks. For same rank, highest crowding.
            # So we should sort by ranks ascending, then crowding descending.
            # np.lexsort(( -crowding_all, ranks_all ))
            
            sort_idx = np.lexsort((-crowding_all, ranks_all))
            
            # Take first N
            next_idx = sort_idx[:N]
            
            X = X_all[next_idx]
            F = F_all[next_idx]
            
            # Check if we still have budget for the next generation's offspring
            # The loop condition handles this.
            
        return X

    def _nsga2_sort(self, X, F, N):
        """
        Performs non-dominated sorting and calculates crowding distance.
        Returns ranks (array of ints) and crowding distances (array of floats).
        """
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n, dtype=float)
        
        # Vectorized dominance check
        # dominates[i, j] is True if i dominates j
        F_expanded_i = F[:, np.newaxis, :]
        F_expanded_j = F[np.newaxis, :, :]
        
        leq = np.all(F_expanded_i <= F_expanded_j, axis=2)
        neq = np.any(F_expanded_i < F_expanded_j, axis=2)
        dominates = leq & neq
        
        # n_dominated[i] is the number of individuals dominated by i
        n_dominated = np.sum(dominates, axis=1)
        
        # Front 0
        front = np.where(n_dominated == 0)[0]
        for i in front:
            ranks[i] = 0
            
        k = 0
        while len(front) > 0:
            next_front = []
            for i in front:
                # Find all j such that i dominates j
                j_indices = np.where(dominates[i, :])[0]
                for j in j_indices:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
                        ranks[j] = k + 1
            front = next_front
            k += 1
        
        # Calculate crowding distance
        for k in range(max(ranks) + 1):
            front_indices = np.where(ranks == k)[0]
            if len(front_indices) <= 2:
                crowding[front_indices] = np.inf
                continue
            
            for m in range(self.n_obj):
                sorted_idx = front_indices[np.argsort(F[front_indices, m])]
                crowding[sorted_idx[0]] = np.inf
                crowding[sorted_idx[-1]] = np.inf
                
                f_min = F[sorted_idx[0], m]
                f_max = F[sorted_idx[-1], m]
                if f_max == f_min:
                    continue
                    
                # Vectorized distance calculation
                f_vals = F[sorted_idx, m]
                dists = np.zeros(len(sorted_idx))
                dists[1:-1] = (f_vals[2:] - f_vals[:-2]) / (f_max - f_min)
                crowding[sorted_idx] += dists
        
        return ranks, crowding

    def _tournament_select(self, ranks, crowding, N, k=2):
        """
        Tournament selection.
        """
        # Select k random individuals
        indices = np.random.choice(N, k, replace=False)
        # Find the best one
        best_idx = 0
        best_rank = ranks[indices[0]]
        best_crowding = crowding[indices[0]]
        
        for i in range(1, k):
            idx = indices[i]
            r = ranks[idx]
            c = crowding[idx]
            # Lower rank is better. If same rank, higher crowding is better.
            if r < best_rank or (r == best_rank and c > best_crowding):
                best_idx = i
                best_rank = r
                best_crowding = c
        
        return indices[best_idx]

    def _sbx_crossover(self, x1, x2, lo, hi, eta_c):
        """
        Simulated Binary Crossover (Vectorized).
        """
        c1 = np.zeros_like(x1)
        c2 = np.zeros_like(x2)
        
        # Vectorize the random decisions
        rand_mask = np.random.rand(len(x1)) < 0.5
        rand_beta = np.random.rand(len(x1))
        
        # Calculate beta for all dimensions
        # beta = (2 * u)^(1/(eta+1)) if u <= 1/(eta+1) else (1/(2*(1-u)))^(1/(eta+1))
        u = rand_beta
        beta = np.where(u <= 1.0 / (eta_c + 1), 
                        (2.0 * u) ** (1.0 / (eta_c + 1)), 
                        (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1)))
        
        # Apply crossover where rand_mask is True and x1 != x2
        diff = np.abs(x1 - x2) > 1e-14
        active = rand_mask & diff
        
        # For active genes, apply SBX
        c1[active] = 0.5 * ((1.0 + beta[active]) * x1[active] + (1.0 - beta[active]) * x2[active])
        c2[active] = 0.5 * ((1.0 - beta[active]) * x1[active] + (1.0 + beta[active]) * x2[active])
        
        # For inactive genes, swap or copy
        # If rand_mask is True but diff is False, they are same, so copy
        # If rand_mask is False, swap
        swap_mask = ~rand_mask
        c1[swap_mask] = x2[swap_mask]
        c2[swap_mask] = x1[swap_mask]
        
        # Handle the case where rand_mask is True but diff is False (x1==x2)
        # c1 should be x1, c2 should be x2
        same_mask = rand_mask & ~diff
        c1[same_mask] = x1[same_mask]
        c2[same_mask] = x2[same_mask]
        
        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        
        return c1, c2

    def _poly_mutation(self, x, lo, hi, pm, eta_m):
        """
        Polynomial Mutation (Vectorized).
        """
        x_mut = x.copy()
        
        # Vectorize mutation decision
        mut_mask = np.random.rand(len(x)) < pm
        
        # Calculate delta1 and delta2
        range_ = hi - lo
        delta1 = (x_mut - lo) / range_
        delta2 = (hi - x_mut) / range_
        
        # Random direction and magnitude
        rand_dir = np.random.rand(len(x)) < 0.5
        rand_u = np.random.rand(len(x))
        
        # Calculate beta for mutation
        # mut_pow depends on direction
        # If dir is 0 (decrease): mut_pow = 2/(eta+1) * (1 + 2*delta1)^(eta+1)
        # If dir is 1 (increase): mut_pow = 2/(eta+1) * (1 + 2*delta2)^(eta+1)
        
        mut_pow_dec = (2.0 / (eta_m + 1.0)) * (1.0 + 2.0 * delta1) ** (eta_m + 1.0)
        mut_pow_inc = (2.0 / (eta_m + 1.0)) * (1.0 + 2.0 * delta2) ** (eta_m + 1.0)
        
        # beta calculation
        # beta = (2*u)^(1/(eta+1)) if 2*u <= mut_pow else (1/(2*(1-u)))^(1/(eta+1))
        
        beta_dec = np.where(2.0 * rand_u <= mut_pow_dec, 
                            (2.0 * rand_u) ** (1.0 / (eta_m + 1.0)), 
                            (1.0 / (2.0 * (1.0 - rand_u))) ** (1.0 / (eta_m + 1.0)))
        
        beta_inc = np.where(2.0 * rand_u <= mut_pow_inc, 
                            (2.0 * rand_u) ** (1.0 / (eta_m + 1.0)), 
                            (1.0 / (2.0 * (1.0 - rand_u))) ** (1.0 / (eta_m + 1.0)))
        
        # Apply mutation
        dec_mask = mut_mask & rand_dir
        inc_mask = mut_mask & ~rand_dir
        
        x_mut[dec_mask] -= range_[dec_mask] * (beta_dec[dec_mask] - 1.0)
        x_mut[inc_mask] += range_[inc_mask] * (beta_inc[inc_mask] - 1.0)
        
        x_mut = np.clip(x_mut, lo, hi)
        return x_mut
