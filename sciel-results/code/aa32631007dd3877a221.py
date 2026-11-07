import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin for 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Parameters
        pop_size = 30
        if self.budget < pop_size * 2:
            pop_size = max(5, self.budget // 2)
            
        eta_c = 20.0
        eta_m = 20.0
        p_crossover = 0.9
        p_mutation = 1.0 / self.dim

        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        evaluations = pop_size
        
        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            fronts = []
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1
            
            front = [i for i in range(n) if domination_count[i] == 0]
            fronts.append(front)
            
            k = 0
            while front:
                next_front = []
                for i in front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                front = next_front
            return fronts

        # Helper: Crowding distance
        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            distances = np.zeros(n)
            for m in range(self.n_obj):
                # Sort front by objective m
                sorted_indices = np.argsort(F[front_indices, m])
                distances[sorted_indices[0]] = np.inf
                distances[sorted_indices[-1]] = np.inf
                
                f_min = F[front_indices[sorted_indices[0]], m]
                f_max = F[front_indices[sorted_indices[-1]], m]
                if f_max - f_min < 1e-10:
                    continue
                    
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    distances[idx] += (F[front_indices[sorted_indices[i+1]], m] - 
                                       F[front_indices[sorted_indices[i-1]], m]) / (f_max - f_min)
            return distances

        # Helper: SBX
        def sbx(x1, x2, lo, hi, eta):
            c1, c2 = x1.copy(), x2.copy()
            for i in range(self.dim):
                if np.random.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        c1[i] = x1[i]
                        c2[i] = x2[i]
                        continue
                    beta = 0.0
                    if x1[i] <= x2[i]:
                        beta = 2 * (x1[i] - lo[i]) / (x2[i] - x1[i] + 1e-14)
                    else:
                        beta = 2 * (hi[i] - x2[i]) / (x2[i] - x1[i] + 1e-14)
                    beta += 1.0
                    alpha = 1.0 + 2.0 * (1.0 - beta)
                    if np.random.random() <= 1.0 / alpha:
                        beta_q = (0.5 * beta * (np.random.random() ** (1.0 / (eta + 1.0))))
                    else:
                        beta_q = (0.5 * (2.0 - beta) * (1.0 / (np.random.random() ** (1.0 / (eta + 1.0)))))
                    
                    c1[i] = 0.5 * (x1[i] + x2[i] - beta_q * (x2[i] - x1[i]))
                    c2[i] = 0.5 * (x1[i] + x2[i] + beta_q * (x2[i] - x1[i]))
                    
                    if c1[i] < lo[i]: c1[i] = lo[i]
                    if c1[i] > hi[i]: c1[i] = hi[i]
                    if c2[i] < lo[i]: c2[i] = lo[i]
                    if c2[i] > hi[i]: c2[i] = hi[i]
            return c1, c2

        # Helper: Polynomial Mutation
        def pm(x, lo, hi, eta):
            for i in range(self.dim):
                if np.random.random() < p_mutation:
                    if x[i] < lo[i] or x[i] > hi[i]:
                        x[i] = (lo[i] + hi[i]) / 2.0
                    else:
                        y = x[i]
                        if y < lo[i]:
                            y = lo[i]
                        elif y > hi[i]:
                            y = hi[i]
                        
                        r = np.random.random()
                        if r <= 0.5:
                            delta1 = (y - lo[i]) / (hi[i] - lo[i] + 1e-14)
                            xq = 1.0 + (1.0 - delta1) * (2.0 * r) ** (1.0 / (eta + 1.0))
                        else:
                            delta2 = (hi[i] - y) / (hi[i] - lo[i] + 1e-14)
                            xq = 1.0 + (1.0 - delta2) * (2.0 * (1.0 - r)) ** (1.0 / (eta + 1.0))
                        
                        x[i] = y + (hi[i] - lo[i]) * (xq - 1.0)
                        if x[i] < lo[i]: x[i] = lo[i]
                        if x[i] > hi[i]: x[i] = hi[i]
            return x

        # Main Loop
        while evaluations < self.budget:
            if time.time() > deadline:
                break
                
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            # We need to create enough offspring to fill the next population
            # Standard NSGA-II creates N offspring.
            # Since we are limited by total evaluations, we should be careful.
            # Let's generate pop_size offspring.
            
            # Select parents using tournament selection
            def select_parents(F, X, n_parents=2):
                parents_idx = []
                for _ in range(n_parents):
                    candidates = np.random.choice(len(F), size=n_parents, replace=False)
                    # Find non-dominated front of candidates
                    # Simplified: just pick the one with best hypervolume contribution or just random if small
                    # For efficiency in small pop, we can just use non-dominated sorting on the whole pop
                    # But tournament is better.
                    # Let's use simple tournament: pick 2, evaluate dominance.
                    idx = np.random.choice(len(F))
                    parents_idx.append(idx)
                return parents_idx

            # To save computation, we'll just do a standard loop
            while len(offspring_X) < pop_size:
                if time.time() > deadline:
                    break
                if evaluations + 1 > self.budget:
                    break
                    
                # Tournament Selection
                t_size = 2
                candidates = np.random.choice(pop_size, size=t_size, replace=False)
                c1_idx, c2_idx = candidates
                # Check dominance
                if np.all(F[c1_idx] <= F[c2_idx]) and np.any(F[c1_idx] < F[c2_idx]):
                    parent1_idx = c1_idx
                elif np.all(F[c2_idx] <= F[c1_idx]) and np.any(F[c2_idx] < F[c1_idx]):
                    parent1_idx = c2_idx
                else:
                    # Non-dominated, pick random
                    parent1_idx = c1_idx if np.random.random() < 0.5 else c2_idx
                    
                candidates = np.random.choice(pop_size, size=t_size, replace=False)
                c1_idx, c2_idx = candidates
                if np.all(F[c1_idx] <= F[c2_idx]) and np.any(F[c1_idx] < F[c2_idx]):
                    parent2_idx = c1_idx
                elif np.all(F[c2_idx] <= F[c1_idx]) and np.any(F[c2_idx] < F[c1_idx]):
                    parent2_idx = c2_idx
                else:
                    parent2_idx = c1_idx if np.random.random() < 0.5 else c2_idx

                p1 = X[parent1_idx]
                p2 = X[parent2_idx]
                
                if np.random.random() < p_crossover:
                    child1, child2 = sbx(p1, p2, lo, hi, eta_c)
                else:
                    child1, child2 = p1.copy(), p2.copy()
                    
                child1 = pm(child1, lo, hi, eta_m)
                child2 = pm(child2, lo, hi, eta_m)
                
                f1 = problem(child1)
                evaluations += 1
                offspring_X.append(child1)
                offspring_F.append(f1)
                
                if len(offspring_X) < pop_size and evaluations < self.budget:
                    f2 = problem(child2)
                    evaluations += 1
                    offspring_X.append(child2)
                    offspring_F.append(f2)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine populations
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            fronts = non_dominated_sort(F_comb)
            
            # Select next population
            X_new = []
            F_new = []
            
            for front in fronts:
                if len(X_new) + len(front) <= pop_size:
                    # Add all
                    for idx in front:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                else:
                    # Need to truncate
                    remaining = pop_size - len(X_new)
                    # Sort front by crowding distance
                    cd = crowding_distance(F_comb, front)
                    sorted_front = np.argsort(-cd)
                    for i in range(remaining):
                        idx = front[sorted_front[i]]
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)

        return X, F
