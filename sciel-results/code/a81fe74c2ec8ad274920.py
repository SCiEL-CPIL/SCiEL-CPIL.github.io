import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = 20
        if self.budget < N:
            X = lo + (hi - lo) * np.random.random((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return X

        G = (self.budget - N) // N
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N

        # NSGA-II Parameters
        eta_c = 15.0  # Distribution index for crossover
        eta_m = 20.0  # Distribution index for mutation
        pc = 0.9      # Probability of crossover
        pm = 1.0 / self.dim  # Probability of mutation per gene

        def fast_non_dominated_sort(F):
            """
            Returns list of fronts, where each front is a list of indices.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated by i
            n_d = [0] * n  # Number of points dominating i
            rank = [0] * n
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1
                if n_d[i] == 0:
                    rank[i] = 0
                    if not fronts:
                        fronts.append([])
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                Q = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = k + 1
                            Q.append(j)
                k += 1
                if Q:
                    fronts.append(Q)
                else:
                    break
            return fronts

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for points in 'indices' based on F.
            Returns dictionary mapping index to distance.
            """
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            dist = {i: 0.0 for i in indices}
            m = len(indices)
            for k in range(self.n_obj):
                # Sort indices by k-th objective
                sorted_indices = sorted(indices, key=lambda i: F[i][k])
                # Set boundary points to inf
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                # Calculate range
                f_min = F[sorted_indices[0]][k]
                f_max = F[sorted_indices[-1]][k]
                if f_max == f_min:
                    continue
                
                for i in range(1, m - 1):
                    idx = sorted_indices[i]
                    prev_idx = sorted_indices[i-1]
                    next_idx = sorted_indices[i+1]
                    dist[idx] += (F[next_idx][k] - F[prev_idx][k]) / (f_max - f_min)
            
            return dist

        def tournament_select(F, rank, crowding, size=2):
            """
            Binary tournament selection.
            """
            i = np.random.randint(len(F))
            j = np.random.randint(len(F))
            
            # Compare rank first
            if rank[i] < rank[j]:
                return i
            elif rank[j] < rank[i]:
                return j
            else:
                # Compare crowding distance
                if crowding[i] > crowding[j]:
                    return i
                elif crowding[j] > crowding[i]:
                    return j
                else:
                    return i # Random

        def sbx(x1, x2, lo, hi, eta):
            """
            Simulated Binary Crossover.
            """
            if np.random.rand() > pc:
                return x1.copy(), x2.copy()
            
            c1 = x1.copy()
            c2 = x2.copy()
            
            for j in range(self.dim):
                if x1[j] == x2[j]:
                    continue
                if np.random.rand() <= 0.5:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (2 * (1 - u)) ** (1.0 / (eta + 1.0))
                    
                    c1[j] = 0.5 * ((1 + beta) * x1[j] + (1 - beta) * x2[j])
                    c2[j] = 0.5 * ((1 - beta) * x1[j] + (1 + beta) * x2[j])
                
                c1[j] = np.clip(c1[j], lo[j], hi[j])
                c2[j] = np.clip(c2[j], lo[j], hi[j])
            
            return c1, c2

        def polynomial_mutation(x, lo, hi, eta):
            """
            Polynomial Mutation.
            """
            x_mut = x.copy()
            for j in range(self.dim):
                if np.random.rand() < pm:
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1.0))
                        x_mut[j] = x[j] + delta * (hi[j] - lo[j])
                    else:
                        delta = (2 * (1 - u)) ** (1.0 / (eta + 1.0))
                        x_mut[j] = x[j] - delta * (hi[j] - lo[j])
                    x_mut[j] = np.clip(x_mut[j], lo[j], hi[j])
            return x_mut

        for gen in range(G):
            # Generate offspring
            X_off = []
            for _ in range(N):
                p1 = tournament_select(F, rank, crowding)
                p2 = tournament_select(F, rank, crowding)
                
                c1, c2 = sbx(X[p1], X[p2], lo, hi, eta_c)
                
                c1 = polynomial_mutation(c1, lo, hi, eta_m)
                c2 = polynomial_mutation(c2, lo, hi, eta_m)
                
                X_off.append(c1)
                X_off.append(c2)
            
            X_off = np.array(X_off[:N]) # Ensure size N
            F_off = np.array([problem(x) for x in X_off])
            used += N
            
            # Combine parents and offspring
            X_all = np.vstack([X, X_off])
            F_all = np.vstack([F, F_off])
            
            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F_all)
            
            # Assign ranks
            rank = [0] * len(F_all)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Select next generation
            X_new = []
            F_new = []
            current_crowding = {}
            
            for front in fronts:
                if len(X_new) + len(front) <= N:
                    # Add all individuals in this front
                    # Need crowding distance for this front
                    cd = crowding_distance(F_all, front)
                    for idx in front:
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                        current_crowding[idx] = cd[idx]
                else:
                    # Need to select some from this front
                    cd = crowding_distance(F_all, front)
                    # Sort by crowding distance descending
                    front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                    needed = N - len(X_new)
                    for idx in front_sorted[:needed]:
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                        current_crowding[idx] = cd[idx]
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)
            
            # Recalculate rank and crowding for the new population for next tournament
            # Note: The rank array from fast_non_dominated_sort on F_all is not directly usable
            # because the indices are from F_all. We need rank for F_new.
            # However, tournament selection in next gen uses F_new.
            # We can just re-run fast_non_dominated_sort on F_new?
            # Or we can map the indices.
            # It's safer and simpler to just re-sort F_new.
            # But F_new is just a subset of F_all.
            # Let's just re-run the sort on F_new to get correct ranks for the new population.
            fronts_new = fast_non_dominated_sort(F)
            rank = [0] * N
            for i, front in enumerate(fronts_new):
                for idx in front:
                    rank[idx] = i
            
            # Calculate crowding for all fronts in new population
            # We need a single crowding dict for the whole population?
            # NSGA-II tournament compares rank first, then crowding distance.
            # Crowding distance is only defined within a front.
            # If ranks are equal, we compare crowding distances.
            # So we need crowding distance for each individual in its respective front.
            crowding = {i: 0.0 for i in range(N)}
            for front in fronts_new:
                if len(front) > 2:
                    cd = crowding_distance(F, front)
                    for idx in front:
                        crowding[idx] = cd[idx]
                else:
                    for idx in front:
                        crowding[idx] = float('inf')

        return X
