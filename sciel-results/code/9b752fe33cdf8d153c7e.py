import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        N = min(100, self.budget)
        
        # Initial population
        pop = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in pop])
        used = N
        
        # Parameters
        eta_c = 15.0  # Crossover distribution index
        eta_m = 20.0  # Mutation distribution index
        prob_c = 0.9  # Crossover probability
        prob_m = 1.0 / self.dim  # Mutation probability per variable

        while used < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Non-dominated sorting
            ranks = self._fast_non_dominated_sort(F)
            
            # 2. Crowding distance
            distances = self._crowding_distance(F, ranks)
            
            # 3. Create mating pool (binary tournament selection)
            # We select N parents
            parents = []
            for _ in range(N):
                i1 = np.random.randint(N)
                i2 = np.random.randint(N)
                if self._tournament_select(i1, i2, ranks, distances):
                    parents.append(pop[i1])
                else:
                    parents.append(pop[i2])
            
            # 4. Generate offspring
            offspring = np.zeros_like(pop)
            for i in range(0, N, 2):
                if np.random.random() < prob_c:
                    offspring[i], offspring[i+1] = self._sbx_crossover(parents[i], parents[i+1], lo, hi, eta_c)
                else:
                    offspring[i] = parents[i]
                    offspring[i+1] = parents[i+1]
                
                # Mutation
                offspring[i] = self._polynomial_mutation(offspring[i], lo, hi, eta_m, prob_m)
                offspring[i+1] = self._polynomial_mutation(offspring[i+1], lo, hi, eta_m, prob_m)
            
            # 5. Evaluate offspring
            F_off = np.array([problem(x) for x in offspring])
            used += N
            
            # 6. Combine parent and offspring populations
            combined_pop = np.vstack([pop, offspring])
            combined_F = np.vstack([F, F_off])
            
            # 7. Selection for next generation
            ranks_comb = self._fast_non_dominated_sort(combined_F)
            distances_comb = self._crowding_distance(combined_F, ranks_comb)
            
            next_pop_idx = []
            front = 0
            while len(next_pop_idx) < N:
                # Get indices in this front
                front_indices = np.where(ranks_comb == front)[0]
                if len(next_pop_idx) + len(front_indices) <= N:
                    next_pop_idx.extend(front_indices)
                    front += 1
                else:
                    # Sort by crowding distance and take best
                    sorted_indices = front_indices[np.argsort(distances_comb[front_indices])[::-1]]
                    next_pop_idx.extend(sorted_indices[:N - len(next_pop_idx)])
                    break
            
            pop = combined_pop[next_pop_idx]
            F = combined_F[next_pop_idx]

        return pop

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]  # Dominated sets
        n_dominates = np.zeros(n, dtype=int)
        ranks = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominates[i] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominates[j] += 1
        
        front = 0
        current_front = [i for i in range(n) if n_dominates[i] == 0]
        
        while current_front:
            next_front = []
            for i in current_front:
                ranks[i] = front
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        next_front.append(j)
            current_front = next_front
            front += 1
            
        return ranks

    def _crowding_distance(self, F, ranks):
        n = len(F)
        distances = np.zeros(n)
        
        # Group by rank
        unique_ranks = np.unique(ranks)
        for r in unique_ranks:
            indices = np.where(ranks == r)[0]
            m = len(indices)
            if m <= 2:
                distances[indices] = np.inf
                continue
            
            for k in range(self.n_obj):
                # Sort by objective k
                sorted_indices = indices[np.argsort(F[indices, k])]
                distances[sorted_indices[0]] = np.inf
                distances[sorted_indices[-1]] = np.inf
                
                f_min = F[sorted_indices[0], k]
                f_max = F[sorted_indices[-1], k]
                
                if f_max - f_min == 0:
                    continue
                
                for i in range(1, m - 1):
                    distances[sorted_indices[i]] += (F[sorted_indices[i+1], k] - F[sorted_indices[i-1], k]) / (f_max - f_min)
        
        return distances

    def _tournament_select(self, i1, i2, ranks, distances):
        # Lower rank is better
        if ranks[i1] != ranks[i2]:
            return ranks[i1] < ranks[i2]
        # Higher distance is better
        return distances[i1] > distances[i2]

    def _sbx_crossover(self, p1, p2, lo, hi, eta):
        c1 = p1.copy()
        c2 = p2.copy()
        
        for i in range(self.dim):
            if np.random.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if p1[i] < p2[i]:
                        beta = 1.0 + 2.0 * (p1[i] - lo[i]) / (p2[i] - p1[i])
                        alpha = 1.0 + 2.0 * (hi[i] - p2[i]) / (p2[i] - p1[i])
                    else:
                        beta = 1.0 + 2.0 * (hi[i] - p2[i]) / (p1[i] - p2[i])
                        alpha = 1.0 + 2.0 * (p1[i] - lo[i]) / (p1[i] - p2[i])
                    
                    rand = np.random.random()
                    if rand <= 1.0 / (beta + 1.0):
                        beta_q = (2.0 * rand * (beta - 1.0) + 1.0) ** (1.0 / (eta + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 * (1.0 - rand) * (alpha - 1.0) + 1.0)) ** (1.0 / (eta + 1.0))
                    
                    c1[i] = 0.5 * (p1[i] + p2[i] - beta_q * (p2[i] - p1[i]))
                    c2[i] = 0.5 * (p1[i] + p2[i] + beta_q * (p2[i] - p1[i]))
                    
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, eta, prob):
        x_mut = x.copy()
        for i in range(self.dim):
            if np.random.random() < prob:
                delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i])
                rand = np.random.random()
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    mut_pow = 2.0 + 1.0 / (eta + 1.0) * (1.0 + xy)
                    u = np.random.random()
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** mut_pow)
                    deltaq = (val - 1.0) ** (1.0 / (eta + 1.0))
                    x_mut[i] += deltaq * (hi[i] - lo[i])
                else:
                    xy = 1.0 - delta2
                    mut_pow = 2.0 + 1.0 / (eta + 1.0) * (1.0 + xy)
                    u = np.random.random()
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** mut_pow)
                    deltaq = (1.0 - val ** (1.0 / (eta + 1.0)))
                    x_mut[i] -= deltaq * (hi[i] - lo[i])
                
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut
