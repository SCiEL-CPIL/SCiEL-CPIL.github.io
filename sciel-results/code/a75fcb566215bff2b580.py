# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Objective Optimizer using NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eta = 20.0  # SBX distribution index
        self.mu = 20.0   # Mutation distribution index

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # NSGA-II Parameters
        pop_size = 40
        n_gen = self.budget // pop_size
        if n_gen < 2:
            n_gen = 2
            pop_size = self.budget // n_gen
            
        # Initialize Population
        X = lo + np.random.rand(pop_size, self.dim) * range_
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # Pre-compute ranks for faster tournament selection in later generations?
        # No, ranks change every generation.
        
        # Main Loop
        for gen in range(n_gen):
            if time.time() > deadline - 5 or evals >= self.budget:
                break
            
            # 1. Fast Non-Dominated Sorting
            frontiers = self._fast_non_dominated_sort(F)
            
            # Compute ranks for each individual
            ranks = np.zeros(pop_size, dtype=int)
            for rank, frontier in enumerate(frontiers):
                ranks[frontier] = rank
            
            # 2. Calculate Crowding Distance
            cd = np.zeros(pop_size)
            for frontier in frontiers:
                if len(frontier) > 0:
                    cd[frontier] = self._crowding_distance(F[frontier])
            
            # 3. Create Mating Pool
            parents = []
            for _ in range(pop_size // 2):
                if evals >= self.budget:
                    break
                p1 = self._tournament_select(ranks, cd)
                p2 = self._tournament_select(ranks, cd)
                parents.append(p1)
                parents.append(p2)
            
            # 4. Crossover and Mutation
            offspring_X = []
            offspring_F = []
            
            for i in range(0, len(parents), 2):
                if evals >= self.budget:
                    break
                p1_idx = parents[i]
                p2_idx = parents[i+1]
                
                # SBX Crossover
                if np.random.rand() < 0.9:
                    c1, c2 = self._sbx_crossover(X[p1_idx], X[p2_idx], lo, hi)
                else:
                    c1, c2 = X[p1_idx].copy(), X[p2_idx].copy()
                
                # Polynomial Mutation
                c1 = self._polynomial_mutation(c1, lo, hi)
                c2 = self._polynomial_mutation(c2, lo, hi)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
                
                f1 = problem(c1)
                f2 = problem(c2)
                evals += 2
                
                offspring_F.append(f1)
                offspring_F.append(f2)
            
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # 5. Combine and Select
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            combined_frontiers = self._fast_non_dominated_sort(combined_F)
            combined_cd = np.zeros(len(combined_F))
            for frontier in combined_frontiers:
                if len(frontier) > 0:
                    combined_cd[frontier] = self._crowding_distance(combined_F[frontier])
            
            # Select next population
            new_indices = []
            for frontier in combined_frontiers:
                if len(new_indices) + len(frontier) <= pop_size:
                    new_indices.extend(frontier)
                else:
                    remaining = pop_size - len(new_indices)
                    # Sort by crowding distance descending
                    sorted_frontier = np.argsort(-combined_cd[frontier])
                    new_indices.extend(frontier[sorted_frontier[:remaining]])
                    break
            
            X = combined_X[new_indices]
            F = combined_F[new_indices]
            
        return X

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        n_count = np.zeros(n, dtype=int)
        frontiers = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_count[i] += 1
        
        current_frontier = []
        for i in range(n):
            if n_count[i] == 0:
                current_frontier.append(i)
        
        while current_frontier:
            frontiers.append(current_frontier)
            next_frontier = []
            for i in current_frontier:
                for j in S[i]:
                    n_count[j] -= 1
                    if n_count[j] == 0:
                        next_frontier.append(j)
            current_frontier = next_frontier
            
        return frontiers

    def _crowding_distance(self, F):
        n = len(F)
        if n == 0:
            return np.array([])
        cd = np.zeros(n)
        if n <= 2:
            cd[:] = float('inf')
            return cd
            
        for m in range(self.n_obj):
            sorted_idx = np.argsort(F[:, m])
            cd[sorted_idx[0]] = float('inf')
            cd[sorted_idx[-1]] = float('inf')
            
            f_min = F[sorted_idx[0], m]
            f_max = F[sorted_idx[-1], m]
            
            if f_max == f_min:
                continue
            
            for i in range(1, n - 1):
                cd[sorted_idx[i]] += (F[sorted_idx[i + 1], m] - F[sorted_idx[i - 1], m]) / (f_max - f_min)
        
        return cd

    def _tournament_select(self, ranks, cd):
        i = np.random.randint(len(ranks))
        j = np.random.randint(len(ranks))
        
        if ranks[i] < ranks[j]:
            return i
        elif ranks[j] < ranks[i]:
            return j
        else:
            if cd[i] > cd[j]:
                return i
            else:
                return j

    def _sbx_crossover(self, x1, x2, lo, hi):
        c1 = x1.copy()
        c2 = x2.copy()
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if x1[i] != x2[i]:
                    if x1[i] > x2[i]:
                        x1_val, x2_val = x2[i], x1[i]
                    else:
                        x1_val, x2_val = x1[i], x2[i]
                    
                    if np.random.rand() < 0.5:
                        beta = 1.0 / (np.random.rand() + 1.0)
                        beta_q = (2.0 * beta ** (self.eta + 1.0)) ** (-1.0)
                        c1[i] = 0.5 * (x1_val + x2_val) - 0.5 * (x2_val - x1_val) * beta_q
                    else:
                        beta = 1.0 / (1.0 - np.random.rand())
                        beta_q = (2.0 * (1.0 - beta) ** (self.eta + 1.0)) ** (-1.0)
                        c1[i] = 0.5 * (x1_val + x2_val) + 0.5 * (x2_val - x1_val) * beta_q
                    
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi):
        for i in range(self.dim):
            if np.random.rand() < 0.1:
                y = x[i]
                if y <= lo[i] or y >= hi[i]:
                    continue
                delta1 = (y - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - y) / (hi[i] - lo[i])
                rand = np.random.rand()
                mut_pow = self.mu
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (mut_pow + 1.0))
                    deltaq = (val ** (1.0 / (mut_pow + 1.0)) - 1.0)
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (mut_pow + 1.0))
                    deltaq = 1.0 - (val ** (1.0 / (mut_pow + 1.0)))
                x[i] = y + deltaq * (hi[i] - lo[i])
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x
# EVOLVE-BLOCK-END
