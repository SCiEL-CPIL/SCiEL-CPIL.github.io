import numpy as np
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        
    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Set random seed for reproducibility within the call if needed, 
        # but the evaluator sets the global seed. 
        # We will use np.random directly.
        
        # NSGA-II Parameters
        pop_size = 50
        n_generations = self.budget // pop_size
        # If budget is not a multiple of pop_size, we might have leftover evaluations.
        # We can just run for n_generations generations.
        # 200 // 50 = 4 generations.
        
        # Initial Population: Use Sobol for better initial spread
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Generate pop_size points
        # Sobol requires power of 2 for best performance, but works for any N.
        # To get exactly pop_size, we can generate a power of 2 >= pop_size and slice, 
        # or just use random uniform if we want to be simple. 
        # Sobol is better.
        n_sobol = 2 ** int(np.ceil(np.log2(pop_size)))
        points = sampler.random(n_sobol)
        points = points[:pop_size]
        
        X = lo + points * (hi - lo)
        F = np.array([problem(x) for x in X])
        
        # Main Loop
        for gen in range(n_generations):
            # Non-dominated sorting and crowding distance
            fronts = self._non_dominated_sort(F)
            crowding_dists = self._crowding_distance(F, fronts)
            
            # Selection: Tournament selection based on rank and crowding distance
            parents = []
            while len(parents) < pop_size:
                p1 = self._tournament_select(F, fronts, crowding_dists)
                p2 = self._tournament_select(F, fronts, crowding_dists)
                parents.append(p1)
                parents.append(p2)
                
            # Crossover and Mutation
            offspring_X = np.zeros((pop_size, self.dim))
            for i in range(0, pop_size, 2):
                if i + 1 < pop_size:
                    x1 = X[parents[i]]
                    x2 = X[parents[i+1]]
                    
                    # Simulated Binary Crossover (SBX)
                    if np.random.rand() < 0.9: # Crossover rate
                        offspring1, offspring2 = self._sbx(x1, x2, lo, hi)
                    else:
                        offspring1 = x1.copy()
                        offspring2 = x2.copy()
                        
                    # Polynomial Mutation
                    offspring1 = self._poly_mutation(offspring1, lo, hi)
                    offspring2 = self._poly_mutation(offspring2, lo, hi)
                    
                    offspring_X[i] = offspring1
                    offspring_X[i+1] = offspring2
            # Handle odd pop_size if necessary, though 50 is even.
            if pop_size % 2 != 0:
                 offspring_X[-1] = X[parents[-1]].copy()
                 offspring_X[-1] = self._poly_mutation(offspring_X[-1], lo, hi)

            # Evaluate Offspring
            F_offspring = np.array([problem(x) for x in offspring_X])
            
            # Combine Parent and Offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, F_offspring])
            
            # Non-dominated sorting on combined population
            fronts_combined = self._non_dominated_sort(F_combined)
            crowding_combined = self._crowding_distance(F_combined, fronts_combined)
            
            # Selection for next generation: Elitism
            # Sort by rank, then by crowding distance (descending)
            # We want to select pop_size individuals.
            selected_indices = []
            for rank in fronts_combined:
                if len(selected_indices) + len(rank) <= pop_size:
                    selected_indices.extend(rank)
                else:
                    # Need to pick only some from this rank
                    # Sort this rank by crowding distance descending
                    rank_indices = np.array(rank)
                    # Get crowding distances for these indices
                    cd = crowding_combined[rank_indices]
                    # Sort indices by cd descending
                    sorted_idx_in_rank = rank_indices[np.argsort(cd)[::-1]]
                    needed = pop_size - len(selected_indices)
                    selected_indices.extend(sorted_idx_in_rank[:needed])
                    break
                    
            X = X_combined[selected_indices]
            F = F_combined[selected_indices]
            
        return None

    def _non_dominated_sort(self, F):
        """
        Perform non-dominated sorting.
        Returns a list of lists, where each inner list contains indices of points in that front.
        """
        n = len(F)
        S = [[] for _ in range(n)] # Dominated set
        n_d = np.zeros(n, dtype=int) # Number of dominators
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_d[j] += 1
                elif self._dominates(F[j], F[i]):
                    S[j].append(i)
                    n_d[i] += 1
                    
        # First front
        current_front = []
        for i in range(n):
            if n_d[i] == 0:
                current_front.append(i)
                
        if not current_front:
            return [[]]
            
        fronts.append(current_front)
        
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_d[j] -= 1
                    if n_d[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front = next_front
            
        return fronts

    def _dominates(self, a, b):
        """
        Check if a dominates b.
        a dominates b if a is better than or equal to b in all objectives
        and strictly better in at least one.
        """
        return np.all(a <= b) and np.any(a < b)

    def _crowding_distance(self, F, fronts):
        """
        Calculate crowding distance for each point.
        """
        n = len(F)
        dists = np.zeros(n)
        
        for front in fronts:
            m = len(front)
            if m <= 2:
                dists[front] = np.inf
                continue
                
            for k in range(self.n_obj):
                # Sort front by k-th objective
                sorted_indices = np.array(front)[np.argsort(F[front, k])]
                
                # Assign infinite distance to best and worst
                dists[sorted_indices[0]] = np.inf
                dists[sorted_indices[-1]] = np.inf
                
                # Normalize objective range
                f_min = F[sorted_indices[0], k]
                f_max = F[sorted_indices[-1], k]
                range_k = f_max - f_min
                
                if range_k == 0:
                    continue
                    
                for i in range(1, m - 1):
                    dists[sorted_indices[i]] += (F[sorted_indices[i+1], k] - F[sorted_indices[i-1], k]) / range_k
                    
        return dists

    def _tournament_select(self, F, fronts, crowding_dists):
        """
        Tournament selection.
        """
        n = len(F)
        # Assign ranks
        ranks = np.zeros(n, dtype=int)
        for i, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = i + 1 # 1-based rank for sorting (lower is better)
                
        # Pick two random candidates
        i = np.random.randint(n)
        j = np.random.randint(n)
        
        # Compare ranks
        if ranks[i] < ranks[j]:
            return i
        elif ranks[j] < ranks[i]:
            return j
        else:
            # Same rank, compare crowding distance
            if crowding_dists[i] > crowding_dists[j]:
                return i
            else:
                return j

    def _sbx(self, x1, x2, lo, hi):
        """
        Simulated Binary Crossover.
        """
        eta = 20.0
        offspring1 = x1.copy()
        offspring2 = x2.copy()
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                continue
                
            if abs(x1[i] - x2[i]) > 1e-14:
                if x1[i] < x2[i]:
                    x_lo, x_hi = x1[i], x2[i]
                else:
                    x_lo, x_hi = x2[i], x1[i]
                    
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    
                c1 = 0.5 * ((x1[i] + x2[i]) - beta * (x2[i] - x1[i]))
                c2 = 0.5 * ((x1[i] + x2[i]) + beta * (x2[i] - x1[i]))
                
                # Clip to bounds
                c1 = np.clip(c1, lo[i], hi[i])
                c2 = np.clip(c2, lo[i], hi[i])
                
                offspring1[i] = c1
                offspring2[i] = c2
            else:
                # Swap if necessary? Standard SBX produces c1, c2. 
                # We just assign them.
                pass
                
        return offspring1, offspring2

    def _poly_mutation(self, x, lo, hi):
        """
        Polynomial Mutation.
        """
        eta_m = 20.0
        x_mut = x.copy()
        
        for i in range(self.dim):
            if np.random.rand() < 0.9: # Mutation rate
                u = np.random.rand()
                if u < 0.5:
                    delta1 = (2 * u) ** (1 / (eta_m + 1)) - 1
                else:
                    delta1 = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                    
                # Calculate range
                range_i = hi[i] - lo[i]
                
                x_new = x[i] + delta1 * range_i
                x_new = np.clip(x_new, lo[i], hi[i])
                x_mut[i] = x_new
                
        return x_mut

    def _pareto_front_indices(self, F):
        """
        Helper to find Pareto front indices (used in old code, kept for compatibility if needed, 
        though NSGA-II uses _non_dominated_sort).
        """
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        return np.where(~is_dominated)[0]
