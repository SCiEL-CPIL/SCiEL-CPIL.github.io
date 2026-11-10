import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        self.eval_count = 0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size: Small enough to allow many generations, large enough for diversity.
        # 200 budget. If pop=20, we can do ~10 generations.
        pop_size = 20
        max_time = 320.0 # Safety margin for 330s limit

        # 1. Initialization
        # Use random initialization within bounds
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        self.eval_count += pop_size

        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]  # Dominated sets
            n_dominates = [0] * n          # Number of solutions dominated by i
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    # j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominates[i] += 1
            
            for i in range(n):
                if n_dominates[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            
            # Remove last empty front
            if not fronts[-1]:
                fronts.pop()
                
            return fronts

        # Helper: Crowding distance
        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            
            dist = np.zeros(n)
            for m in range(n_obj):
                # Sort front by m-th objective
                sorted_indices = np.argsort([F[idx][m] for idx in front_indices])
                # Assign boundary points infinity
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                f_min = F[front_indices[sorted_indices[0]]][m]
                f_max = F[front_indices[sorted_indices[-1]]][m]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                    
                for i in range(1, n - 1):
                    idx_prev = front_indices[sorted_indices[i-1]]
                    idx_next = front_indices[sorted_indices[i+1]]
                    dist[sorted_indices[i]] += (F[idx_next][m] - F[idx_prev][m]) / f_range
            
            return dist

        # Main Loop
        while self.eval_count < self.budget:
            # Check time
            if time.time() - self.start_time > max_time:
                break

            # Determine how many offspring we can generate
            # We need to evaluate offspring. Let's generate pop_size offspring each generation.
            # If remaining budget < pop_size, generate only remaining amount?
            # NSGA-II usually keeps population size constant.
            # If budget is 200 and pop is 20, we do 10 generations.
            
            offspring_size = min(pop_size, self.budget - self.eval_count)
            if offspring_size <= 0:
                break

            # Create offspring using Differential Evolution (DE/rand/1/bin)
            # Select parents from non-dominated fronts
            fronts = non_dominated_sort(F)
            # Prefer parents from front 0, then 1, etc.
            parent_pool = []
            for front in fronts:
                parent_pool.extend(front)
                if len(parent_pool) >= pop_size:
                    break
            
            # If we don't have enough parents (unlikely), fill with random
            while len(parent_pool) < pop_size:
                parent_pool.append(np.random.randint(pop_size))

            Q = []
            FQ = []
            
            for _ in range(offspring_size):
                # Select 3 distinct parents
                p1_idx, p2_idx, p3_idx = np.random.choice(len(parent_pool), 3, replace=False)
                p1 = X[parent_pool[p1_idx]]
                p2 = X[parent_pool[p2_idx]]
                p3 = X[parent_pool[p3_idx]]
                
                # DE Mutation: v = p1 + F * (p2 - p3)
                F_factor = 0.8 # Standard DE factor
                v = p1 + F_factor * (p2 - p3)
                
                # Crossover: Binomial
                CR = 0.9
                trial = v.copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = p1[j]
                        
                # Clamp to bounds
                trial = np.clip(trial, lo, hi)
                
                # Evaluate
                f_trial = problem(trial)
                self.eval_count += 1
                Q.append(trial)
                FQ.append(f_trial)
                
                # Time check inside loop to be safe
                if time.time() - self.start_time > max_time:
                    break

            if not Q:
                break

            Q = np.array(Q)
            FQ = np.array(FQ)
            
            # Combine Population and Offspring
            R_X = np.vstack([X, Q])
            R_F = np.vstack([F, FQ])
            
            # Non-dominated sorting on combined population
            fronts = non_dominated_sort(R_F)
            
            # Select next population P
            new_P = []
            new_PF = []
            
            for front in fronts:
                if len(new_P) + len(front) <= pop_size:
                    # Add entire front
                    new_P.extend(front)
                else:
                    # Need to truncate front using crowding distance
                    front_F = R_F[front]
                    cd = crowding_distance(front_F, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    needed = pop_size - len(new_P)
                    new_P.extend(sorted_front[:needed])
                    break
            
            if len(new_P) == 0:
                break

            X = R_X[new_P]
            F = R_F[new_P]

        return X
