import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit, 330s kill
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # With budget 200, N=20 allows ~9 generations.
        N = 20
        
        rng = np.random.default_rng(0)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Helper: Fast Non-Dominated Sort
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_doms = [0] * n
            fronts = [[]]
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_doms[p] += 1
            
            for p in range(n):
                if n_doms[p] == 0:
                    fronts[0].append(p)
            
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_doms[q] -= 1
                        if n_doms[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            
            return fronts[:-1] # Remove last empty list
        
        # Helper: Crowding Distance
        def crowding_distance(F, indices):
            if not indices:
                return {}
            m = len(indices)
            if m <= 2:
                return {i: float('inf') for i in indices}
            
            cd = {i: 0.0 for i in indices}
            for k in range(n_obj):
                f_vals = np.array([F[i][k] for i in indices])
                min_f, max_f = np.min(f_vals), np.max(f_vals)
                if max_f == min_f:
                    for i in indices:
                        cd[i] = float('inf')
                    continue
                indices_sorted = sorted(indices, key=lambda i: F[i][k])
                cd[indices_sorted[0]] = float('inf')
                cd[indices_sorted[-1]] = float('inf')
                for j in range(1, m - 1):
                    i_prev = indices_sorted[j-1]
                    i_next = indices_sorted[j+1]
                    i_curr = indices_sorted[j]
                    cd[i_curr] += (F[i_next][k] - F[i_prev][k]) / (max_f - min_f)
            return cd
        
        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, beta=20):
            x1 = x1.astype(float)
            x2 = x2.astype(float)
            x3 = x1.copy()
            x4 = x2.copy()
            
            mask = rng.random(dim) < 0.5
            for i in range(dim):
                if not mask[i]:
                    continue
                if abs(x1[i] - x2[i]) < 1e-14:
                    x3[i] = x1[i]
                    x4[i] = x2[i]
                    continue
                u = rng.random()
                if u <= 0.5:
                    alpha = 2.0 * u
                    beta_c = (alpha + 1.0) ** (1.0 / (beta + 1.0))
                else:
                    alpha = 2.0 * (1.0 - u)
                    beta_c = (alpha + 1.0) ** (1.0 / (beta + 1.0))
                
                if x1[i] < x2[i]:
                    x3[i] = 0.5 * ((1.0 + beta_c) * x1[i] + (1.0 - beta_c) * x2[i])
                    x4[i] = 0.5 * ((1.0 - beta_c) * x1[i] + (1.0 + beta_c) * x2[i])
                else:
                    x3[i] = 0.5 * ((1.0 + beta_c) * x2[i] + (1.0 - beta_c) * x1[i])
                    x4[i] = 0.5 * ((1.0 - beta_c) * x2[i] + (1.0 + beta_c) * x1[i])
                    
            return x3, x4
            
        # Helper: Polynomial Mutation
        def poly_mutation(x, eta=20):
            x = x.copy()
            for i in range(dim):
                if rng.random() < 1.0 / dim:
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta_q = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    x[i] += delta_q * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        while evals < budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            for _ in range(N):
                # Tournament selection
                p1_idx = rng.integers(N)
                p2_idx = rng.integers(N)
                
                # Need ranks and crowding for selection
                # We recompute ranks for current population X
                # Optimization: Cache ranks if possible, but N is small, recompute is fine
                fronts = fast_non_dominated_sort(F)
                ranks = np.zeros(N, dtype=int)
                for r, front in enumerate(fronts):
                    for idx in front:
                        ranks[idx] = r
                
                cd = {}
                for front in fronts:
                    cd_front = crowding_distance(F, front)
                    cd.update(cd_front)
                
                def better(a, b):
                    if ranks[a] != ranks[b]:
                        return ranks[a] < ranks[b]
                    return cd.get(a, 0) > cd.get(b, 0)
                
                p1 = p1_idx if better(p1_idx, p2_idx) else p2_idx
                
                # Crossover
                x1, x2 = X[p1], X[rng.integers(N)]
                c1, c2 = sbx_crossover(x1, x2)
                
                # Mutation
                c1 = poly_mutation(c1)
                
                offspring_X.append(c1)
                fc = problem(c1)
                offspring_F.append(fc)
                evals += 1
                if evals >= budget:
                    break
            
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine
            U_X = np.vstack([X, offspring_X])
            U_F = np.vstack([F, offspring_F])
            
            # Sort and Select
            fronts = fast_non_dominated_sort(U_F)
            next_X = []
            next_F = []
            
            for front in fronts:
                if len(next_X) + len(front) <= N:
                    for idx in front:
                        next_X.append(U_X[idx])
                        next_F.append(U_F[idx])
                else:
                    # Need to select subset of this front
                    cd_front = crowding_distance(U_F, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd_front[i], reverse=True)
                    needed = N - len(next_X)
                    for idx in sorted_front[:needed]:
                        next_X.append(U_X[idx])
                        next_F.append(U_F[idx])
                    break
                    
            if len(next_X) == 0:
                break
                
            X = np.array(next_X)
            F = np.array(next_F)
            
        return X
