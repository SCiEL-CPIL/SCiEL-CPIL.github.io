import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        pop_size = 20
        n_gen = self.budget // pop_size
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II parameters
        crossover_prob = 0.9
        mutation_prob = 0.1
        eta_c = 20.0  # Crossover distribution index
        eta_m = 20.0  # Mutation distribution index
        
        for gen in range(n_gen):
            # Non-dominated sorting
            ranks, fronts = self._non_dominated_sort(F)
            
            # Crowding distance
            crowding = self._crowding_distance(F, fronts)
            
            # Select parents
            parents = []
            while len(parents) < pop_size:
                # Binary tournament selection
                i1, i2 = rng.integers(0, pop_size, 2)
                if ranks[i1] < ranks[i2] or (ranks[i1] == ranks[i2] and crowding[i1] > crowding[i2]):
                    parents.append(X[i1])
                else:
                    parents.append(X[i2])
            
            # Create offspring
            offspring = []
            for _ in range(pop_size // 2):
                # Select two parents
                p1, p2 = parents[rng.integers(0, pop_size, 2)]
                
                # Crossover
                if rng.random() < crossover_prob:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Mutation
                c1 = self._polynomial_mutation(c1, lo, hi, eta_m, mutation_prob, self.dim, rng)
                c2 = self._polynomial_mutation(c2, lo, hi, eta_m, mutation_prob, self.dim, rng)
                
                offspring.append(c1)
                offspring.append(c2)
            
            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring])
            used += pop_size
            
            # Combine parent and offspring populations
            X_combined = np.vstack([X, offspring])
            F_combined = np.vstack([F, F_off])
            
            # Non-dominated sorting on combined population
            ranks_combined, fronts_combined = self._non_dominated_sort(F_combined)
            
            # Select next generation
            next_gen = []
            next_gen_F = []
            for front in fronts_combined:
                if len(next_gen) + len(front) <= pop_size:
                    next_gen.extend(X_combined[front])
                    next_gen_F.extend(F_combined[front])
                else:
                    # Sort by crowding distance in descending order
                    crowding_front = self._crowding_distance(F_combined[front], [front])
                    sorted_indices = np.argsort(-crowding_front)
                    needed = pop_size - len(next_gen)
                    selected = [front[i] for i in sorted_indices[:needed]]
                    next_gen.extend(X_combined[selected])
                    next_gen_F.extend(F_combined[selected])
                    break
            
            X = np.array(next_gen)
            F = np.array(next_gen_F)
        
        return X
    
    def _non_dominated_sort(self, F):
        """Perform non-dominated sorting and return ranks and fronts."""
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        fronts = []
        S = [[] for _ in range(n)]  # S[i] = set of points dominated by i
        n_dominated = np.zeros(n, dtype=int)  # Number of points dominated by i
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominated[i] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominated[j] += 1
        
        current_front = [i for i in range(n) if n_dominated[i] == 0]
        front_idx = 0
        while current_front:
            fronts.append(current_front)
            for i in current_front:
                ranks[i] = front_idx
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
            current_front = next_front
            front_idx += 1
        
        return ranks, fronts
    
    def _crowding_distance(self, F, fronts):
        """Calculate crowding distance for each point."""
        n = len(F)
        crowding = np.zeros(n)
        
        for front in fronts:
            if len(front) <= 2:
                crowding[front] = np.inf
                continue
            
            for k in range(F.shape[1]):
                sorted_front = sorted(front, key=lambda i: F[i][k])
                crowding[sorted_front[0]] = np.inf
                crowding[sorted_front[-1]] = np.inf
                
                f_min = F[sorted_front[0]][k]
                f_max = F[sorted_front[-1]][k]
                
                if f_max - f_min < 1e-12:
                    continue
                
                for i in range(1, len(sorted_front) - 1):
                    prev_idx = sorted_front[i - 1]
                    next_idx = sorted_front[i + 1]
                    crowding[sorted_front[i]] += (F[next_idx][k] - F[prev_idx][k]) / (f_max - f_min)
        
        return crowding
    
    def _sbx_crossover(self, p1, p2, lo, hi, eta, rng):
        """Simulated Binary Crossover."""
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p2)
        
        for i in range(self.dim):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) < 1e-12:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
                else:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
            else:
                c1[i] = p1[i]
                c2[i] = p2[i]
        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        
        return c1, c2
    
    def _polynomial_mutation(self, x, lo, hi, eta, prob, dim, rng):
        """Polynomial Mutation."""
        x_mut = x.copy()
        for i in range(dim):
            if rng.random() < prob:
                if x_mut[i] <= lo[i] or x_mut[i] >= hi[i]:
                    continue
                delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i])
                u = rng.random()
                if u < 0.5:
                    xy = (1 + delta1) ** (eta + 1)
                    deltaq = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    xy = (1 + delta2) ** (eta + 1)
                    deltaq = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                x_mut[i] += deltaq * (hi[i] - lo[i])
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        
        return x_mut
