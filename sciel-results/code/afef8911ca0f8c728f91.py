import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        
        # Population size
        pop_size = 40
        
        # 1. Initialization using LHS
        # Simple LHS: divide range into pop_size intervals, pick one random point in each for each dim
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            # Create bins
            bins = np.linspace(lo[d], hi[d], pop_size + 1)
            # Random permutation of bins
            perm = np.random.permutation(pop_size)
            for i in range(pop_size):
                # Pick a random point in the perm[i]-th bin
                low_bin = bins[perm[i]]
                high_bin = bins[perm[i] + 1]
                X[i, d] = np.random.uniform(low_bin, high_bin)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts[0].append(i)
            i = 0
            while len(fronts[i]) > 0:
                Q = []
                for idx in fronts[i]:
                    for j in S[idx]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            Q.append(j)
                if Q:
                    i += 1
                    fronts.append(Q)
                else:
                    break
            return fronts

        # Helper: Crowding distance
        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for k in range(F.shape[1]):
                # Sort indices by objective k
                sorted_idx = np.argsort(F[indices, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    f_curr = F[indices[sorted_idx[i]], k]
                    f_prev = F[indices[sorted_idx[i-1]], k]
                    f_next = F[indices[sorted_idx[i+1]], k]
                    range_k = F[indices, k].max() - F[indices, k].min()
                    if range_k > 0:
                        cd[sorted_idx[i]] += (f_next - f_prev) / range_k
            return cd

        # Helper: SBX Crossover
        def sbx_crossover(p1, p2, eta=20.0):
            if np.random.rand() < 0.5:
                return p1.copy(), p2.copy()
            
            c1 = np.zeros_like(p1)
            c2 = np.zeros_like(p2)
            
            for i in range(len(p1)):
                if np.random.rand() < 0.5:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
                    continue
                
                if p1[i] == p2[i]:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
                else:
                    u = np.random.rand()
                    if u <= 0.5:
                        alpha = 2.0 * u
                        beta = ((alpha + 1.0) ** (1.0 / (eta + 1.0)))
                        c1[i] = 0.5 * ((p1[i] + p2[i]) - beta * (p1[i] - p2[i]))
                        c2[i] = 0.5 * ((p1[i] + p2[i]) + beta * (p1[i] - p2[i]))
                    else:
                        alpha = 2.0 * (1.0 - u)
                        beta = ((alpha + 1.0) ** (1.0 / (eta + 1.0)))
                        c1[i] = 0.5 * ((p1[i] + p2[i]) + beta * (p1[i] - p2[i]))
                        c2[i] = 0.5 * ((p1[i] + p2[i]) - beta * (p1[i] - p2[i]))
            
            return c1, c2

        # Helper: Polynomial Mutation
        def poly_mutation(x, eta=20.0):
            x_mut = x.copy()
            for i in range(len(x)):
                if np.random.rand() < 0.5:
                    u = np.random.rand()
                    if u <= 0.5:
                        delta_q = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta_q = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    
                    x_mut[i] = x[i] + delta_q * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Main Loop
        while used < budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            offX = []
            offF = []
            
            # We generate pop_size offspring
            for i in range(pop_size):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Tournament Selection
                def tournament_select(pop_X, pop_F, k=2):
                    idxs = np.random.choice(len(pop_X), k, replace=False)
                    best_idx = idxs[0]
                    best_F = pop_F[best_idx]
                    for idx in idxs[1:]:
                        if np.any(pop_F[idx] < best_F):
                            # Simple domination check for tournament is tricky without full sort
                            # We just pick the one with smallest sum of objectives as a heuristic for tournament
                            # Or better, use the fronts computed in previous step? 
                            # To save time, we just use a simple heuristic: minimize sum of objectives
                            pass
                    # Actually, let's just pick the index with the minimum sum of objectives among the k
                    sums = np.sum(pop_F[idxs], axis=1)
                    return idxs[np.argmin(sums)]

                # Better selection: Use the fronts from the previous generation
                # We need to maintain fronts
                # Let's restructure: compute fronts, then select parents based on fronts
                
                # For simplicity and speed in this constrained budget, 
                # we will select parents randomly from the current population 
                # but weighted by their rank in the previous sort?
                # No, let's just do standard NSGA-II selection properly.
                
                # We need fronts of the *current* population X, F
                # This is expensive if done every generation for every parent.
                # Let's compute fronts once per generation.
                
                pass # Placeholder, logic below
                
            # Recompute fronts for current population
            fronts = non_dominated_sort(F)
            
            # Assign ranks
            ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Crowding distance for the last front? 
            # Actually, we need to select parents.
            # We'll use a simple approach: 
            # 1. Compute fronts.
            # 2. For selection, use tournament on rank, then crowding distance.
            
            # To avoid O(N^2) crowding distance calculation for every selection, 
            # we will just use rank for selection and rely on SBX/Mutation for diversity.
            # Or we can compute crowding distance for the whole population once per generation.
            
            # Let's compute crowding distance for the whole population
            # This is O(N^2 * M). N=40, M=3. 1600*3 = 4800 ops. Very fast.
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 2:
                    cd_front = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd[idx] = cd_front[i]
                else:
                    for idx in front:
                        cd[idx] = np.inf

            # Now generate offspring
            offX = []
            offF = []
            
            for i in range(pop_size):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Tournament Selection
                def tournament_select():
                    idxs = np.random.choice(len(X), 2, replace=False)
                    # Compare ranks
                    if ranks[idxs[0]] < ranks[idxs[1]]:
                        return idxs[0]
                    elif ranks[idxs[1]] < ranks[idxs[0]]:
                        return idxs[1]
                    else:
                        # Tie-breaker: Crowding distance
                        if cd[idxs[0]] > cd[idxs[1]]:
                            return idxs[0]
                        else:
                            return idxs[1]

                p1_idx = tournament_select()
                p2_idx = tournament_select()
                
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                c1, c2 = sbx_crossover(p1, p2, eta=20.0)
                c1 = poly_mutation(c1, eta=20.0)
                c2 = poly_mutation(c2, eta=20.0)
                
                offX.append(c1)
                offF.append(problem(c1))
                used += 1
                
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                offX.append(c2)
                offF.append(problem(c2))
                used += 1
                
            if not offX:
                break
                
            offX = np.array(offX)
            offF = np.array(offF)
            
            # Combine and Select
            totalX = np.vstack([X, offX])
            totalF = np.vstack([F, offF])
            
            # Sort total population
            total_fronts = non_dominated_sort(totalF)
            
            keep_indices = []
            for front in total_fronts:
                if len(keep_indices) + len(front) <= pop_size:
                    keep_indices.extend(front)
                else:
                    need = pop_size - len(keep_indices)
                    if need > 0:
                        # Select based on crowding distance
                        cd_front = crowding_distance(totalF, front)
                        # Sort by crowding distance descending
                        sorted_front = sorted(front, key=lambda i: cd_front[front.index(i)], reverse=True)
                        keep_indices.extend(sorted_front[:need])
                    break
                    
            X = totalX[keep_indices]
            F = totalF[keep_indices]
            
        return X
