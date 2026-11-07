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
        n = self.dim
        m = self.n_obj
        budget = self.budget
        
        # NSGA-II Parameters
        pop_size = 50
        crossover_eta = 20.0
        mutation_eta = 20.0
        
        # Initialize
        X = lo + rng.random((pop_size, n)) * (hi - lo)
        F = np.array([problem(x) for x in X], dtype=float)
        evals = pop_size
        
        # Helper: Fast Non-Dominated Sorting
        def fast_non_dominated_sort(F):
            N = len(F)
            S = [[] for _ in range(N)]
            n_count = np.zeros(N, dtype=int)
            rank = np.zeros(N, dtype=int)
            
            for p in range(N):
                for q in range(N):
                    if p == q:
                        continue
                    # Check if p dominates q
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_count[p] += 1
                        
                if n_count[p] == 0:
                    rank[p] = 0
                    
            i = 0
            front = [0]
            while front:
                next_front = []
                for p in front:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                i += 1
                front = next_front
                
            return rank, S

        # Helper: Crowding Distance
        def crowding_distance(F, rank, N):
            cd = np.zeros(N)
            fronts = {}
            for i in range(N):
                if rank[i] not in fronts:
                    fronts[rank[i]] = []
                fronts[rank[i]].append(i)
                
            for r, members in fronts.items():
                if len(members) <= 2:
                    for idx in members:
                        cd[idx] = float('inf')
                    continue
                for k in range(m):
                    # Sort by objective k
                    sorted_indices = sorted(members, key=lambda idx: F[idx][k])
                    cd[sorted_indices[0]] = float('inf')
                    cd[sorted_indices[-1]] = float('inf')
                    f_min = F[sorted_indices[0]][k]
                    f_max = F[sorted_indices[-1]][k]
                    if f_max == f_min:
                        continue
                    for j in range(1, len(sorted_indices) - 1):
                        idx_prev = sorted_indices[j-1]
                        idx_curr = sorted_indices[j]
                        idx_next = sorted_indices[j+1]
                        cd[idx_curr] += (F[idx_next][k] - F[idx_prev][k]) / (f_max - f_min)
            return cd

        # Helper: Tournament Selection
        def tournament_select(rank, cd, size=2):
            candidates = rng.integers(0, len(rank), size=size)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best]:
                    best = c
                elif rank[c] == rank[best] and cd[c] > cd[best]:
                    best = c
            return best

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2):
            if rng.random() > 0.9: # Crossover rate 0.9? Let's use 1.0 for small pop
                return x1.copy(), x2.copy()
            
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for i in range(n):
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (crossover_eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1.0 / (crossover_eta + 1))
                
                c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                
                child1[i] = c1
                child2[i] = c2
            return child1, child2

        # Helper: Polynomial Mutation
        def polynomial_mutation(x):
            for i in range(n):
                if rng.random() < 1.0 / n:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (mutation_eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (mutation_eta + 1))
                    x[i] = x[i] + delta * (hi[i] - lo[i])
            return np.clip(x, lo, hi)

        # Main Loop
        while evals < budget:
            Y = np.zeros((pop_size, n))
            for i in range(pop_size):
                if evals >= budget:
                    break
                
                # Select parents
                p1_idx = tournament_select(rank, cd)
                p2_idx = tournament_select(rank, cd)
                
                # Crossover
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                
                # Mutation
                c1 = polynomial_mutation(c1)
                
                Y[i] = c1
                F_child = problem(c1)
                evals += 1
                F[i] = F_child # Temporary store, will be overwritten in merge step logic below?
                # Actually, we need to store F for Y separately or just evaluate and store
                # Let's store in a list for Y
                # Re-evaluating structure:
                # We need F_Y corresponding to Y
            
            # Corrected Loop Structure for NSGA-II
            
        # The above loop was messy. Let's rewrite the main part cleanly.
        
        # Reset for clean implementation
        X = lo + rng.random((pop_size, n)) * (hi - lo)
        F = np.array([problem(x) for x in X], dtype=float)
        evals = pop_size
        
        rank, S = fast_non_dominated_sort(F)
        cd = crowding_distance(F, rank, pop_size)
        
        while evals < budget:
            Y = np.zeros((pop_size, n))
            F_Y = np.zeros((pop_size, m))
            
            for i in range(pop_size):
                if evals >= budget:
                    break
                
                p1_idx = tournament_select(rank, cd)
                p2_idx = tournament_select(rank, cd)
                
                c1, _ = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = polynomial_mutation(c1)
                
                Y[i] = c1
                F_Y[i] = problem(c1)
                evals += 1
            
            # Merge
            X_total = np.vstack([X, Y])
            F_total = np.vstack([F, F_Y])
            
            rank_total, S_total = fast_non_dominated_sort(F_total)
            cd_total = crowding_distance(F_total, rank_total, 2 * pop_size)
            
            # Select next population
            selected_indices = np.zeros(pop_size, dtype=int)
            count = 0
            for r in range(max(rank_total) + 1):
                members = [i for i in range(2 * pop_size) if rank_total[i] == r]
                if not members:
                    continue
                if count + len(members) <= pop_size:
                    for idx in members:
                        selected_indices[count] = idx
                        count += 1
                else:
                    # Sort by crowding distance
                    members_sorted = sorted(members, key=lambda i: cd_total[i], reverse=True)
                    for idx in members_sorted:
                        if count < pop_size:
                            selected_indices[count] = idx
                            count += 1
                        else:
                            break
                if count >= pop_size:
                    break
            
            X = X_total[selected_indices]
            F = F_total[selected_indices]
            
            rank, S = fast_non_dominated_sort(F)
            cd = crowding_distance(F, rank, pop_size)

        return X
