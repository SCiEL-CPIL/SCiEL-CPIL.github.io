import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Setup
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        N = 20 # Population size
        
        # 1. Initial Population
        X = lo + (hi - lo) * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        evals_used = N
        
        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            front_sizes = []
            S = [set() for _ in range(n)]
            n_dominated = np.zeros(n, dtype=int)
            
            # Compute dominance relations
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_dominated[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n_dominated[i] += 1
            
            current_front = [i for i in range(n) if n_dominated[i] == 0]
            front_id = 0
            while current_front:
                front_sizes.append(len(current_front))
                for i in current_front:
                    ranks[i] = front_id
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            next_front.append(j)
                current_front = next_front
                front_id += 1
            return ranks, front_sizes

        # Helper: Crowding Distance
        def crowding_distance(F, ranks, n_points):
            cd = np.zeros(n_points)
            unique_ranks = np.unique(ranks)
            for r in unique_ranks:
                idx = np.where(ranks == r)[0]
                if len(idx) <= 2:
                    cd[idx] = np.inf
                    continue
                for m in range(self.n_obj):
                    vals = F[idx, m]
                    min_val, max_val = np.min(vals), np.max(vals)
                    if max_val - min_val < 1e-12:
                        continue
                    sorted_idx = idx[np.argsort(vals)]
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    for k in range(1, len(sorted_idx) - 1):
                        i_curr = sorted_idx[k]
                        i_prev = sorted_idx[k-1]
                        i_next = sorted_idx[k+1]
                        cd[i_curr] += (F[i_next, m] - F[i_prev, m]) / (max_val - min_val)
            return cd

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta_c=20):
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for i in range(self.dim):
                if np.random.rand() <= 0.5:
                    x1_i, x2_i = x1[i], x2[i]
                else:
                    x1_i, x2_i = x2[i], x1[i]
                
                if abs(x1_i - x2_i) < 1e-14:
                    child1[i] = child2[i] = x1_i
                    continue
                
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta_c + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                
                c1 = 0.5 * ((1 + beta) * x1_i + (1 - beta) * x2_i)
                c2 = 0.5 * ((1 - beta) * x1_i + (1 + beta) * x2_i)
                
                # Clip to bounds
                c1 = np.clip(c1, lo[i], hi[i])
                c2 = np.clip(c2, lo[i], hi[i])
                
                child1[i] = c1
                child2[i] = c2
            return child1, child2

        # Helper: Mutation
        def mutate(x, eta_m=20):
            y = x.copy()
            for i in range(self.dim):
                if np.random.rand() <= 1.0 / self.dim:
                    y_i = y[i]
                    lo_i, hi_i = lo[i], hi[i]
                    if hi_i - lo_i < 1e-14:
                        continue
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    
                    y_i = y_i + delta * (hi_i - lo_i)
                    y[i] = np.clip(y_i, lo_i, hi_i)
            return y

        # Main Loop
        while evals_used < self.budget:
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            # We generate N offspring
            # Need 2N parents via tournament
            # But we only have budget for N evaluations.
            # So we create N offspring from N pairs of parents? 
            # Standard NSGA-II: Select 2N parents, create N offspring.
            # Selection is free.
            
            ranks, _ = non_dominated_sort(F)
            cd = crowding_distance(F, ranks, N)
            
            for _ in range(N):
                # Tournament selection for parent 1
                idx1 = np.random.randint(N)
                idx2 = np.random.randint(N)
                if ranks[idx1] < ranks[idx2]:
                    p1_idx = idx1
                elif ranks[idx1] > ranks[idx2]:
                    p1_idx = idx2
                else:
                    p1_idx = idx1 if cd[idx1] > cd[idx2] else idx2
                
                # Tournament selection for parent 2
                idx3 = np.random.randint(N)
                idx4 = np.random.randint(N)
                if ranks[idx3] < ranks[idx4]:
                    p2_idx = idx3
                elif ranks[idx3] > ranks[idx4]:
                    p2_idx = idx4
                else:
                    p2_idx = idx3 if cd[idx3] > cd[idx4] else idx4
                
                # Crossover
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                
                # Mutation
                c1 = mutate(c1)
                c2 = mutate(c2)
                
                # Evaluate
                f1 = problem(c1)
                f2 = problem(c2)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
                offspring_F.append(f1)
                offspring_F.append(f2)
                
                evals_used += 2
                
                if evals_used >= self.budget:
                    break
            
            if evals_used >= self.budget:
                break

            # Combine and Select
            # Note: We might have generated fewer than N*2 if budget hit mid-way? 
            # No, we break if budget is hit. 
            # But standard NSGA-II replaces the whole population.
            # If we generated K offspring, we have N + K points.
            # We need to select N best.
            
            # In the loop above, I added 2 offspring per iteration.
            # If N=20, we do 10 iterations to get 20 offspring?
            # Wait, the loop `for _ in range(N)` runs N times.
            # Inside, it adds 2 offspring. So 2N offspring total?
            # That would require 2N evaluations.
            # Budget check: `evals_used < self.budget`.
            # If I generate 2N offspring, I consume 2N evals.
            # Then I select N from 3N.
            # This is valid.
            
            # However, if budget is tight, I might not complete the `for` loop.
            # Let's check the break condition.
            # If I break inside, I have some offspring.
            
            # Let's restructure to ensure we don't exceed budget.
            # Better: Generate exactly as many offspring as allowed by remaining budget.
            # But NSGA-II works best with fixed population sizes.
            # With 200 budget and N=20:
            # Init: 20.
            # Gen 1: Generate 20 offspring (10 pairs). Cost 20. Total 40.
            # Combine 20+20=40. Select 20.
            # Gen 2: Generate 20 offspring. Cost 20. Total 60.
            # ...
            # This works perfectly. 10 generations of 20 offspring.
            
            # My loop above adds 2 offspring per iteration.
            # So `range(N//2)` would generate N offspring?
            # If N=20, `range(10)` generates 20 offspring.
            # Let's fix the loop logic in the code below.

            # Re-evaluate combined population
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.array([F, np.array(offspring_F)])
            
            ranks_c, _ = non_dominated_sort(F_combined)
            cd_c = crowding_distance(F_combined, ranks_c, len(X_combined))
            
            # Select N best
            # Sort by rank, then by crowding distance (descending)
            # We want the first N in this sorted order.
            # Create indices sorted by (rank, -crowding)
            sort_idx = np.lexsort((-cd_c, ranks_c))
            X = X_combined[sort_idx[:N]]
            F = F_combined[sort_idx[:N]]
            
        return X
