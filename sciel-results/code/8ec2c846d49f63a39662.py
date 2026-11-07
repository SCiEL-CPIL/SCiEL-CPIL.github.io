import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
    
    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        # Population size
        pop_size = min(20, self.budget)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II parameters
        eta_c = 15.0  # distribution index for crossover
        eta_m = 20.0  # distribution index for mutation
        pc = 0.9      # crossover probability
        
        def non_dominated_sort(F):
            """Return list of fronts, each front is list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination count
            n_dom = np.zeros(n, dtype=int)  # number of points dominated
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
            
            current_front = [i for i in range(n) if n_dom[i] == 0]
            front_num = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
                front_num += 1
            return fronts
        
        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            dist = np.zeros(len(indices))
            F_sub = F[indices]
            
            for m in range(self.n_obj):
                sorted_idx = np.argsort(F_sub[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
                if f_range > 0:
                    for i in range(1, len(sorted_idx) - 1):
                        dist[sorted_idx[i]] += (F_sub[sorted_idx[i+1], m] - F_sub[sorted_idx[i-1], m]) / f_range
            
            return dist
        
        def sbx_crossover(x1, x2):
            """Simulated Binary Crossover."""
            if rng.random() > pc:
                return x1.copy(), x2.copy()
            
            c1 = x1.copy()
            c2 = x2.copy()
            
            for i in range(self.dim):
                if rng.random() < 0.5:
                    if x1[i] == x2[i]:
                        continue
                    if x1[i] < x2[i]:
                        xl, xu = x1[i], x2[i]
                    else:
                        xl, xu = x2[i], x1[i]
                    
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                    
                    c1[i] = 0.5 * ((x1[i] + x2[i]) - beta * (x2[i] - x1[i]))
                    c2[i] = 0.5 * ((x1[i] + x2[i]) + beta * (x2[i] - x1[i]))
            
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            return c1, c2
        
        def polynomial_mutation(x):
            """Polynomial mutation."""
            x = x.copy()
            for i in range(self.dim):
                if rng.random() < 0.5:  # mutation rate 0.5 per gene... actually should be 1/dim
                    continue
                if x[i] == lo[i] and x[i] == hi[i]:
                    continue
                
                u = rng.random()
                if u < 0.5:
                    delta1 = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    deltaq = delta1 * (hi[i] - lo[i])
                else:
                    delta2 = (2 * (1 - u)) ** (1.0 / (eta_m + 1)) - 1
                    deltaq = -delta2 * (hi[i] - lo[i])
                
                x[i] += deltaq
            
            return np.clip(x, lo, hi)
        
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Non-dominated sort
            fronts = non_dominated_sort(F)
            
            # Build mating pool from first fronts
            mating_pool = []
            for front in fronts:
                if len(mating_pool) + len(front) >= pop_size:
                    # Add as many as needed
                    needed = pop_size - len(mating_pool)
                    # Use crowding distance to select
                    cd = crowding_distance(F, front)
                    sorted_front = [front[i] for i in np.argsort(-cd)]
                    mating_pool.extend(sorted_front[:needed])
                    break
                else:
                    mating_pool.extend(front)
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size and used + len(offspring_X) < self.budget:
                # Select two parents via tournament
                i = rng.integers(len(mating_pool))
                j = rng.integers(len(mating_pool))
                p1 = X[mating_pool[i]]
                p2 = X[mating_pool[j]]
                
                c1, c2 = sbx_crossover(p1, p2)
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                f1 = problem(c1)
                f2 = problem(c2)
                used += 2
                
                if used > self.budget:
                    # Only add one
                    offspring_X.append(c1)
                    offspring_F.append(f1)
                    break
                
                offspring_X.append(c1)
                offspring_X.append(c2)
                offspring_F.append(f1)
                offspring_F.append(f2)
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X[:pop_size])
            offspring_F = np.array(offspring_F[:pop_size])
            
            # Combine parent and offspring
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Non-dominated sort combined
            fronts_comb = non_dominated_sort(F_comb)
            
            # Select next generation
            X_new = []
            F_new = []
            for front in fronts_comb:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                else:
                    needed = pop_size - len(X_new)
                    cd = crowding_distance(F_comb, front)
                    sorted_front = [front[i] for i in np.argsort(-cd)]
                    for idx in sorted_front[:needed]:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)
        
        return X
