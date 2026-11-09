import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 280.0  # Safety margin

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Population size
        # With budget 200, N=20 gives 10 generations (1 init + 9 offspring)
        N = 20
        if budget < N:
            N = budget

        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # n_dom[i] = number of individuals dominating i
            fronts = [[]]
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
                    # If neither dominates, do nothing
                    
            # First front
            for i in range(n):
                if n_dom[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
                
            # Remove empty fronts at the end if any
            while fronts and not fronts[-1]:
                fronts.pop()
                
            return fronts

        # Helper: Crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            n = len(indices)
            dist = np.zeros(n)
            F_sub = F[indices]
            
            for m in range(F_sub.shape[1]):
                # Sort by objective m
                sorted_idx = np.argsort(F_sub[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                
                f_min = F_sub[sorted_idx[0], m]
                f_max = F_sub[sorted_idx[-1], m]
                
                if f_max > f_min:
                    for i in range(1, n - 1):
                        dist[sorted_idx[i]] += (F_sub[sorted_idx[i + 1], m] - F_sub[sorted_idx[i - 1], m]) / (f_max - f_min)
                else:
                    # If all values are the same, distance is 0 for all? Or inf?
                    # Usually 0 if range is 0, but to avoid division by zero, we skip.
                    pass
                    
            return dist

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta=15.0, p_c=0.9):
            if rng.random() > p_c:
                return x1.copy(), x2.copy()
            
            c1 = x1.copy()
            c2 = x2.copy()
            
            for i in range(dim):
                if x1[i] == x2[i]:
                    continue
                
                if rng.random() < 0.5:
                    if 2.0 * rng.random() <= (1.0 + 1.0 / (eta + 1.0)):
                        beta = (2.0 * rng.random()) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - rng.random()))) ** (1.0 / (eta + 1.0))
                    
                    c1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                    c2[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
                else:
                    if 2.0 * rng.random() <= (1.0 + 1.0 / (eta + 1.0)):
                        beta = (2.0 * rng.random()) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - rng.random()))) ** (1.0 / (eta + 1.0))
                    
                    c1[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
                    c2[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                    
            return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

        # Helper: Polynomial Mutation
        def polynomial_mutation(x, eta=20.0, p_m=0.2):
            for i in range(dim):
                if rng.random() < p_m:
                    if rng.random() < 0.5:
                        delta = (2.0 * rng.random()) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2.0 * rng.random()) ** (1.0 / (eta + 1.0))
                    
                    x[i] += delta * (hi[i] - lo[i])
            return np.clip(x, lo, hi)

        # Main Loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Create offspring
            offspring_X = np.zeros((N, dim))
            
            for i in range(0, N, 2):
                # Select parents using tournament selection on current X, F
                # We need to select from the current population X, F
                
                # Find non-dominated fronts of current population
                fronts = non_dominated_sort(F)
                
                # Assign rank
                rank = np.zeros(N, dtype=int)
                for idx, front in enumerate(fronts):
                    for ind in front:
                        rank[ind] = idx
                
                # Calculate crowding distance for each front
                cd = np.zeros(N)
                for front in fronts:
                    if len(front) > 2:
                        cd_sub = crowding_distance(F, front)
                        for k, ind in enumerate(front):
                            cd[ind] = cd_sub[k]
                    else:
                        for ind in front:
                            cd[ind] = np.inf

                # Tournament selection
                def select_one():
                    i1 = rng.integers(N)
                    i2 = rng.integers(N)
                    if rank[i1] < rank[i2]:
                        return i1
                    elif rank[i2] < rank[i1]:
                        return i2
                    else:
                        if cd[i1] > cd[i2]:
                            return i1
                        else:
                            return i2

                p1 = select_one()
                p2 = select_one()
                
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                offspring_X[i] = c1
                offspring_X[i+1] = c2
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += N
            
            # Combine parents and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])
            
            # Non-dominated sorting on combined population
            fronts = non_dominated_sort(F_all)
            
            # Select next population
            next_X = np.zeros((N, dim))
            next_F = np.zeros((N, n_obj))
            next_idx = 0
            
            for front in fronts:
                if next_idx + len(front) <= N:
                    # Add all
                    for ind in front:
                        next_X[next_idx] = X_all[ind]
                        next_F[next_idx] = F_all[ind]
                        next_idx += 1
                else:
                    # Need to select some
                    k = N - next_idx
                    # Calculate crowding distance for this front
                    cd_sub = crowding_distance(F_all, front)
                    # Sort by crowding distance descending
                    sorted_local = np.argsort(cd_sub)[::-1]
                    for j in range(k):
                        ind = front[sorted_local[j]]
                        next_X[next_idx] = X_all[ind]
                        next_F[next_idx] = F_all[ind]
                        next_idx += 1
                    break
                
                if next_idx >= N:
                    break
            
            X = next_X
            F = next_F
            
        return X
