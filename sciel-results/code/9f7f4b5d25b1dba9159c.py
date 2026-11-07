import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = 50
        if self.budget <= pop_size:
            pop_size = self.budget
            
        # 1. Initialize Population
        # Use Latin Hypercube Sampling for better spread
        X = np.zeros((pop_size, self.dim))
        for i in range(self.dim):
            perm = np.random.permutation(pop_size)
            X[perm, i] = np.linspace(0, 1, pop_size)
        X = lo + X * (hi - lo)
        
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            
            S = [set() for _ in range(n)]
            n_dominated = [0] * n
            ranks = [0] * n
            fronts = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_dominated[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n_dominated[i] += 1
                        # Note: The above logic needs to be symmetric and careful.
                        # Standard NSGA-II:
                        # if i dominates j: S[i].add(j), n_dominated[j]++
                        # if j dominates i: S[j].add(i), n_dominated[i]++
            
            # Re-implementing properly to avoid confusion
            S = [set() for _ in range(n)]
            n_dominated = [0] * n
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
            
            for i in range(n):
                for j in S[i]:
                    n_dominated[j] += 1
            
            current_front = []
            for i in range(n):
                if n_dominated[i] == 0:
                    current_front.append(i)
                    ranks[i] = 0
            if current_front:
                fronts.append(current_front)
            
            k = 0
            while current_front:
                k += 1
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            next_front.append(j)
                            ranks[j] = k
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            
            return fronts

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            n_obj = self.n_obj
            n = len(indices)
            dist = {i: 0.0 for i in indices}
            
            for m in range(n_obj):
                # Sort indices by objective m
                sorted_indices = sorted(indices, key=lambda i: F[i, m])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0], m]
                f_max = F[sorted_indices[-1], m]
                f_range = f_max - f_min
                
                if f_range < 1e-10:
                    continue
                    
                for i in range(1, n - 1):
                    dist[sorted_indices[i]] += (F[sorted_indices[i+1], m] - F[sorted_indices[i-1], m]) / f_range
            
            return dist

        # 2. Evolutionary Loop
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size:
                # Selection: Tournament of 2
                idx1 = np.random.randint(pop_size)
                idx2 = np.random.randint(pop_size)
                
                # We need ranks and crowding distances for the current population
                # Calculating these every step is expensive. 
                # For simplicity and speed with small pop, we can recalculate or cache.
                # Let's recalculate ranks and CD for the current X, F.
                
                # Optimization: Calculate ranks and CD once per generation? 
                # In standard NSGA-II, we select from the current population to create offspring.
                # The current population is fixed during the offspring creation phase.
                
                if not hasattr(self, '_current_ranks') or self._current_F is not F:
                    fronts = non_dominated_sort(F)
                    ranks = np.zeros(pop_size, dtype=int)
                    for rank, front in enumerate(fronts):
                        for idx in front:
                            ranks[idx] = rank
                    
                    # Calculate crowding distance for each front
                    cd = np.zeros(pop_size)
                    for front in fronts:
                        cd_dict = crowding_distance(F, front)
                        for idx in front:
                            cd[idx] = cd_dict[idx]
                    
                    self._current_ranks = ranks
                    self._current_cd = cd
                    self._current_F = F
                    self._current_fronts = fronts
                
                ranks = self._current_ranks
                cd = self._current_cd
                
                # Compare idx1 and idx2
                if ranks[idx1] < ranks[idx2]:
                    p1 = X[idx1]
                elif ranks[idx2] < ranks[idx1]:
                    p1 = X[idx2]
                else:
                    if cd[idx1] > cd[idx2]:
                        p1 = X[idx1]
                    else:
                        p1 = X[idx2]
                
                # Second parent
                idx3 = np.random.randint(pop_size)
                idx4 = np.random.randint(pop_size)
                
                if ranks[idx3] < ranks[idx4]:
                    p2 = X[idx3]
                elif ranks[idx4] < ranks[idx3]:
                    p2 = X[idx4]
                else:
                    if cd[idx3] > cd[idx4]:
                        p2 = X[idx3]
                    else:
                        p2 = X[idx4]
                
                # Crossover (SBX)
                child = np.zeros_like(p1)
                if np.random.rand() < 0.9:
                    for i in range(self.dim):
                        if abs(p1[i] - p2[i]) < 1e-14:
                            child[i] = p1[i]
                        else:
                            u = np.random.rand()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / 21.0)
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / 21.0)
                            if np.random.rand() < 0.5:
                                child[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                            else:
                                child[i] = 0.5 * ((1 + beta) * p2[i] + (1 - beta) * p1[i])
                else:
                    child = p1.copy()
                    
                # Mutation
                for i in range(self.dim):
                    if np.random.rand() < 1.0 / self.dim:
                        u = np.random.rand()
                        delta1 = (1 + 2 * (child[i] - lo[i]) / (hi[i] - lo[i])) ** 21.0
                        delta2 = (1 + 2 * (hi[i] - child[i]) / (hi[i] - lo[i])) ** 21.0
                        if u <= 0.5:
                            deltaq = 1 / delta1
                        else:
                            deltaq = 1 - 1 / delta2
                        child[i] = child[i] + (hi[i] - lo[i]) * (2 * u - 1) * deltaq
                        
                child = np.clip(child, lo, hi)
                
                if evals >= self.budget or time.time() > deadline:
                    break
                    
                fc = problem(child)
                evals += 1
                offspring_X.append(child)
                offspring_F.append(fc)
                
            if not offspring_X:
                break
                
            # Combine
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            
            # Select next generation
            fronts = non_dominated_sort(combined_F)
            
            new_X = []
            new_F = []
            
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    cd_dict = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd_dict[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
                    
            if not new_X:
                break
                
            X = np.array(new_X)
            F = np.array(new_F)
            
            # Clear cached ranks/CD as population changed
            self._current_F = None

        # 3. Return Non-Dominated Set
        # Find non-dominated points in final F
        nd_indices = []
        for i in range(len(F)):
            dominated = False
            for j in range(len(F)):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                nd_indices.append(i)
                
        if not nd_indices:
            nd_indices = list(range(len(F)))
            
        result_X = X[nd_indices]
        return result_X
