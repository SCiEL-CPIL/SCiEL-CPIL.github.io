import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42) # Fixed seed for reproducibility within the run, but evaluator sets np.random.seed
        # The evaluator calls np.random.seed(seed) before creating Optimizer.
        # We should use np.random directly or default_rng. 
        # Using default_rng with a fixed seed might be okay, but to be safe and utilize the evaluator's seed, 
        # let's use np.random functions which respect the global seed set by the evaluator.
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # Budget 200. 
        # If N=20, we can do 10 generations (20 init + 9*20 offspring = 200? No, 20 + 10*20 = 220).
        # Let's do 20 init, then 9 generations of 20 offspring = 20 + 180 = 200.
        N = 20
        
        # Initialize
        X = lo + (hi - lo) * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        
        used = N
        
        # Helper: Fast Non-Dominated Sort
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_d = [0] * n
            fronts = [set()]
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_d[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[j] += 1
                        
                if n_d[i] == 0:
                    fronts[0].add(i)
            
            k = 0
            while len(fronts[k]) > 0:
                next_front = set()
                for i in fronts[k]:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            next_front.add(j)
                k += 1
                fronts.append(next_front)
                
            # Remove the last empty set if any
            if len(fronts) > 0 and len(fronts[-1]) == 0:
                fronts.pop()
                
            return fronts

        # Helper: Crowding Distance
        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n == 0:
                return np.array([])
            if n == 1:
                return np.array([np.inf])
            
            distances = np.zeros(n)
            for m in range(self.n_obj):
                min_val = np.min(F[front_indices, m])
                max_val = np.max(F[front_indices, m])
                if max_val == min_val:
                    continue
                # Sort indices by m-th objective
                sorted_indices = np.argsort(F[front_indices, m])
                # End points have infinite distance
                distances[sorted_indices[0]] = np.inf
                distances[sorted_indices[-1]] = np.inf
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    prev_idx = sorted_indices[i-1]
                    next_idx = sorted_indices[i+1]
                    distances[idx] += (F[front_indices[next_idx], m] - F[front_indices[prev_idx], m]) / (max_val - min_val)
            
            return distances

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta=20.0):
            child1 = np.copy(x1)
            child2 = np.copy(x2)
            mask = np.random.rand(self.dim) < 0.5
            for i in range(self.dim):
                if mask[i]:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1.0))
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return child1, child2

        # Helper: Polynomial Mutation
        def polynomial_mutation(x, eta=20.0):
            x_mut = np.copy(x)
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1.0)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1.0))
                    x_mut[i] += delta * (hi[i] - lo[i])
            return np.clip(x_mut, lo, hi)

        # Main Loop
        while used < self.budget:
            # Generate Offspring
            Q = []
            F_Q = []
            
            # Create N offspring
            for _ in range(N):
                # Tournament Selection
                def tournament_select(X, F, size=2):
                    indices = np.random.choice(len(X), size, replace=False)
                    best_idx = None
                    best_rank = float('inf')
                    best_dist = -float('inf')
                    
                    # We need ranks and distances for selection
                    # To avoid recalculating for every selection, we should calculate once per generation
                    # But for simplicity in this loop structure, let's calculate fronts once before generating offspring
                    pass
                # Actually, we need fronts/dists of current P to select parents.
                # Let's restructure:
                # 1. Calculate fronts and distances for current P
                # 2. Select parents
                # 3. Generate offspring
                # 4. Evaluate offspring
                # 5. Combine, Sort, Select next P
                
                pass # Placeholder

            # Restructuring the loop body properly:
            
            # 1. Sort current P
            fronts = fast_non_dominated_sort(F)
            # Calculate crowding distance for each front
            crowd_dists = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    dists = crowding_distance(F, list(front))
                    for idx, dist in zip(front, dists):
                        crowd_dists[idx] = dist
                        
            # 2. Generate N offspring
            Q_X = []
            for _ in range(N):
                # Select two parents using tournament
                def select_parent():
                    # Tournament size 2
                    i, j = np.random.choice(len(X), 2, replace=False)
                    # Compare (rank, -crowding)
                    rank_i = fronts.index(list(front for front in fronts if i in front)[0]) # This is slow. 
                    # Better: store rank per individual
                    pass
                
                # Let's precompute rank array
                rank_arr = np.zeros(len(X), dtype=int)
                for rank, front in enumerate(fronts):
                    for idx in front:
                        rank_arr[idx] = rank
                
                def select_parent():
                    i, j = np.random.choice(len(X), 2, replace=False)
                    # Lower rank is better
                    if rank_arr[i] < rank_arr[j]:
                        return i
                    elif rank_arr[j] < rank_arr[i]:
                        return j
                    else:
                        # Higher crowding distance is better
                        if crowd_dists[i] > crowd_dists[j]:
                            return i
                        else:
                            return j

                p1 = select_parent()
                p2 = select_parent()
                
                if np.random.rand() < 0.9: # Crossover probability
                    c1, c2 = sbx_crossover(X[p1], X[p2])
                    c1 = polynomial_mutation(c1)
                    c2 = polynomial_mutation(c2)
                    Q_X.append(c1)
                    if len(Q_X) < N:
                        Q_X.append(c2)
                else:
                    c1 = polynomial_mutation(X[p1])
                    Q_X.append(c1)
                    if len(Q_X) < N:
                        Q_X.append(polynomial_mutation(X[p2]))
                        
            # Ensure we have exactly N offspring
            while len(Q_X) < N:
                Q_X.append(polynomial_mutation(X[np.random.randint(len(X))]))
            Q_X = np.array(Q_X[:N])
            
            # 3. Evaluate Offspring
            F_Q = np.array([problem(x) for x in Q_X])
            used += N
            
            # 4. Combine
            R_X = np.vstack([X, Q_X])
            R_F = np.vstack([F, F_Q])
            
            # 5. Sort Combined Population
            R_fronts = fast_non_dominated_sort(R_F)
            
            # 6. Select Next Generation P'
            new_X = []
            new_F = []
            for front in R_fronts:
                if len(new_X) + len(front) <= N:
                    for idx in front:
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                else:
                    # Need to pick only a few from this front based on crowding
                    front_list = list(front)
                    dists = crowding_distance(R_F, front_list)
                    # Sort by distance descending
                    sorted_idx = np.argsort(dists)[::-1]
                    needed = N - len(new_X)
                    for k in range(needed):
                        idx = front_list[sorted_idx[k]]
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                    break
                    
            X = np.array(new_X)
            F = np.array(new_F)
            
        return X
