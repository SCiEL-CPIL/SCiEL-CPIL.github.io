import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0 # 30s margin from 310s/330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = 20
        
        # 1. Initialization
        # Use random initialization
        X = lo + (hi - lo) * np.random.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper functions for NSGA-II
        
        def dominates(a, b):
            # a dominates b if a <= b in all objectives and a < b in at least one
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = np.zeros(n, dtype=int)
            ranks = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_dominates[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        n_dominates[i] += 1
            
            front = 0
            current_front = [i for i in range(n) if n_dominates[i] == 0]
            
            while current_front:
                for i in current_front:
                    ranks[i] = front
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                current_front = next_front
                front += 1
                
            return ranks

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            m = len(indices)
            dist = np.zeros(m)
            F_sub = F[indices]
            
            for k in range(n_obj):
                min_val = np.min(F_sub[:, k])
                max_val = np.max(F_sub[:, k])
                if max_val - min_val < 1e-12:
                    continue
                order = np.argsort(F_sub[:, k])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for i in range(1, m - 1):
                    dist[order[i]] += (F_sub[order[i+1], k] - F_sub[order[i-1], k]) / (max_val - min_val)
                    
            return dist

        def tournament_select(F, ranks, crowding, n_select):
            parents = []
            for _ in range(n_select):
                idx1 = np.random.randint(len(F))
                idx2 = np.random.randint(len(F))
                
                # Compare rank first
                if ranks[idx1] < ranks[idx2]:
                    parents.append(idx1)
                elif ranks[idx2] < ranks[idx1]:
                    parents.append(idx2)
                else:
                    # If same rank, compare crowding distance
                    if crowding[idx1] > crowding[idx2]:
                        parents.append(idx1)
                    else:
                        parents.append(idx2)
            return parents

        def sbx_crossover(x1, x2, lo, hi):
            # Simulated Binary Crossover
            # eta_c = 20
            eta_c = 20
            x = np.zeros_like(x1)
            for i in range(dim):
                if np.random.random() < 0.5:
                    if abs(x2[i] - x1[i]) < 1e-14:
                        x[i] = x1[i]
                    else:
                        if x1[i] < x2[i]:
                            u = np.random.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta_c + 1.0))
                            else:
                                beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
                            c1 = 0.5 * (x1[i] + x2[i] - beta * (x2[i] - x1[i]))
                            c2 = 0.5 * (x1[i] + x2[i] + beta * (x2[i] - x1[i]))
                        else:
                            u = np.random.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta_c + 1.0))
                            else:
                                beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
                            c1 = 0.5 * (x1[i] + x2[i] - beta * (x1[i] - x2[i]))
                            c2 = 0.5 * (x1[i] + x2[i] + beta * (x1[i] - x2[i]))
                        
                        # Clip to bounds
                        c1 = np.clip(c1, lo[i], hi[i])
                        c2 = np.clip(c2, lo[i], hi[i])
                        # Swap if necessary to ensure diversity? Standard SBX just returns c1, c2
                        # We need one child per pair usually, or 2 children.
                        # Here we generate 1 child per pair to keep population size constant?
                        # No, we generate N offspring from N parents.
                        # We pick 2 parents, create 1 child? Or 2?
                        # Standard NSGA-II: Pick 2 parents, create 2 children.
                        # But we need N children. So we pick N/2 pairs?
                        # Let's stick to: For each of N offspring slots, pick 2 parents via tournament, crossover, mutate.
                        # This is simpler.
                        x[i] = c1 if np.random.random() < 0.5 else c2
                else:
                    x[i] = x1[i]
            return x

        def polynomial_mutate(x, lo, hi):
            # eta_m = 20
            eta_m = 20
            for i in range(dim):
                if np.random.random() < 0.5:
                    u = np.random.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                    
                    x[i] += delta * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Main Loop
        while used + N <= budget:
            if time.time() > deadline:
                break
                
            # Evaluate current population for sorting
            ranks = fast_non_dominated_sort(F)
            crowding = crowding_distance(F, np.arange(N))
            
            # Generate offspring
            X_off = np.zeros((N, dim))
            for i in range(N):
                # Select 2 parents
                p1_idx = tournament_select(F, ranks, crowding, 1)[0]
                p2_idx = tournament_select(F, ranks, crowding, 1)[0]
                
                # Crossover
                child = sbx_crossover(X[p1_idx], X[p2_idx], lo, hi)
                
                # Mutation
                child = polynomial_mutate(child, lo, hi)
                
                X_off[i] = child
            
            # Evaluate offspring
            F_off = np.array([problem(x) for x in X_off])
            used += N
            
            # Combine
            X_comb = np.vstack((X, X_off))
            F_comb = np.vstack((F, F_off))
            
            # Sort combined population
            ranks_comb = fast_non_dominated_sort(F_comb)
            
            # Select best N
            # We need to select N points from 2N based on rank and crowding
            # Sort indices by rank, then by crowding distance (descending)
            indices = np.arange(2 * N)
            
            # We can just sort by rank, then for each rank, sort by crowding
            # But we need to pick exactly N.
            
            # Create a list of (rank, crowding, index)
            crowding_comb = crowding_distance(F_comb, np.arange(2 * N))
            
            # Sort by rank asc, then crowding desc
            sort_keys = np.lexsort(( -crowding_comb, ranks_comb))
            
            # Pick first N
            selected_indices = sort_keys[:N]
            
            X = X_comb[selected_indices]
            F = F_comb[selected_indices]

        return X
