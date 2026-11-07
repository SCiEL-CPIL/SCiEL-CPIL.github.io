import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        n = self.dim
        m = self.n_obj
        budget = self.budget
        
        # NSGA-II Parameters
        # Small budget, so keep population small.
        # N=20 allows ~9 generations of offspring after init.
        N = 20
        
        # Initialize population
        X = lo + span * rng.random((N, n))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N
        
        # Helper functions for NSGA-II
        
        def non_dominated_sort(F):
            """
            Returns ranks for each individual.
            F is (N, m).
            """
            N_pop = F.shape[0]
            ranks = np.zeros(N_pop, dtype=int)
            S = [[] for _ in range(N_pop)] # S[i] = indices of individuals dominated by i
            n_dominating = np.zeros(N_pop, dtype=int) # n[i] = number of individuals dominating i
            
            for i in range(N_pop):
                for j in range(i + 1, N_pop):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominating[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominating[i] += 1
            
            current_front = [i for i in range(N_pop) if n_dominating[i] == 0]
            rank = 0
            while current_front:
                next_front = []
                for i in current_front:
                    ranks[i] = rank
                    for j in S[i]:
                        n_dominating[j] -= 1
                        if n_dominating[j] == 0:
                            next_front.append(j)
                current_front = next_front
                rank += 1
            return ranks

        def crowding_distance(F, ranks):
            """
            Returns crowding distance for each individual.
            """
            N_pop = F.shape[0]
            dist = np.zeros(N_pop)
            
            # Group by rank
            for r in range(int(ranks.max()) + 1):
                front_indices = [i for i in range(N_pop) if ranks[i] == r]
                if len(front_indices) <= 2:
                    for i in front_indices:
                        dist[i] = np.inf
                    continue
                
                for k in range(F.shape[1]):
                    # Sort front by objective k
                    front_sorted = sorted(front_indices, key=lambda i: F[i, k])
                    dist[front_sorted[0]] = np.inf
                    dist[front_sorted[-1]] = np.inf
                    f_min = F[front_sorted[0], k]
                    f_max = F[front_sorted[-1], k]
                    if f_max - f_min == 0:
                        continue
                    for idx in range(1, len(front_sorted) - 1):
                        i = front_sorted[idx]
                        i_prev = front_sorted[idx - 1]
                        i_next = front_sorted[idx + 1]
                        dist[i] += (F[i_next, k] - F[i_prev, k]) / (f_max - f_min)
            return dist

        def tournament_select(F, ranks, dist, count=2):
            """
            Selects one individual via binary tournament.
            """
            idx1 = rng.integers(0, N)
            idx2 = rng.integers(0, N)
            if ranks[idx1] < ranks[idx2]:
                return idx1
            elif ranks[idx2] < ranks[idx1]:
                return idx2
            else:
                if dist[idx1] > dist[idx2]:
                    return idx1
                else:
                    return idx2

        def sbx_crossover(x1, x2):
            """
            Simulated Binary Crossover.
            """
            eta = 20.0
            child1 = np.zeros(n)
            child2 = np.zeros(n)
            for k in range(n):
                if rng.random() < 0.5:
                    # Swap
                    pass # We handle both children in loop or just return one? 
                    # Standard SBX produces 2 children. 
                    # For simplicity in this loop structure, let's produce 2 children per pair.
                    # But the loop structure is "for i in range(N): select 2 parents, produce 1 offspring".
                    # This is slightly inefficient for SBX which produces 2.
                    # Let's stick to 1 offspring per iteration to match budget strictly.
                    # We can just take one of the two SBX outputs or use a simpler crossover.
                    # Let's use BLX-alpha or simple arithmetic to ensure 1 child per eval.
                    # Actually, we can do SBX and just pick one child.
                    # Or use Polynomial Mutation on a single parent (mutation only)?
                    # No, crossover is important.
                    # Let's use BLX-Alpha which naturally produces 1 child.
                    pass
            # Let's replace SBX with BLX-Alpha for 1:1 mapping
            return None # Placeholder

        # Let's implement a simpler crossover: BLX-Alpha
        def blx_crossover(x1, x2, alpha=0.5):
            child = np.zeros(n)
            for k in range(n):
                a = min(x1[k], x2[k])
                b = max(x1[k], x2[k])
                if a == b:
                    child[k] = a
                else:
                    # Sample from [a - alpha*(b-a), b + alpha*(b-a)]
                    child[k] = rng.uniform(a - alpha * (b - a), b + alpha * (b - a))
            return child

        def polynomial_mutation(x, eta=20.0):
            # Probability of mutation per gene
            p_mut = 1.0 / n
            for k in range(n):
                if rng.random() < p_mut:
                    y = x[k]
                    l = lo[k]
                    u = hi[k]
                    if y <= l or y >= u:
                        continue
                    delta1 = (y - l) / (u - l)
                    delta2 = (u - y) / (u - l)
                    rand = rng.random()
                    if rand < 0.5:
                        xq = 2 * rand + (1 - 2 * rand) * (delta1 ** (eta + 1))
                    else:
                        xq = 2 * (1 - rand) + (2 * rand - 1) * (delta2 ** (eta + 1))
                    xq = xq - 1
                    if rand < 0.5:
                        deltaq = -(u - l) * xq
                    else:
                        deltaq = (u - l) * xq
                    x[k] = y + deltaq
            return np.clip(x, lo, hi)

        # Main Loop
        while used < budget:
            # Calculate ranks and crowding distance for current population
            ranks = non_dominated_sort(F)
            dist = crowding_distance(F, ranks)
            
            # Generate offspring
            new_X = np.zeros((N, n))
            new_F = np.zeros((N, m))
            
            for i in range(N):
                if used >= budget:
                    break
                    
                # Select parents
                p1 = tournament_select(F, ranks, dist)
                p2 = tournament_select(F, ranks, dist)
                
                # Crossover
                child = blx_crossover(X[p1], X[p2], alpha=0.5)
                
                # Mutation
                child = polynomial_mutation(child, eta=20.0)
                
                # Evaluate
                f_child = problem(child)
                used += 1
                
                new_X[i] = child
                new_F[i] = f_child
            
            # Environmental Selection
            # Merge parents and offspring
            X_all = np.vstack([X, new_X])
            F_all = np.vstack([F, new_F])
            
            # Sort and select top N
            ranks_all = non_dominated_sort(F_all)
            dist_all = crowding_distance(F_all, ranks_all)
            
            # Create list of indices sorted by rank, then crowding distance
            indices = list(range(2 * N))
            indices.sort(key=lambda i: (ranks_all[i], -dist_all[i]))
            
            # Select top N
            selected_indices = indices[:N]
            X = X_all[selected_indices]
            F = F_all[selected_indices]
            
        return X
