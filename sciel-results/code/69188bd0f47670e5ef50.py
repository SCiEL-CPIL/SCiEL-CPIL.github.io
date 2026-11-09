"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Fixed seed for reproducibility as per evaluator description (seed 0 and 1 handled by evaluator's np.random.seed, 
        # but we should use a local rng or respect global if intended. The prompt says "evaluator calls np.random.seed(seed)".
        # The current code uses np.random.default_rng(0). I will use np.random.default_rng(0) to be consistent with the "seed program" 
        # behavior unless the evaluator's seed matters for the internal rng. 
        # Actually, the prompt says: "For seeds 0 and 1 the evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # If I use np.random.default_rng(0), my results are deterministic regardless of the evaluator's seed. 
        # If I use np.random (global), my results depend on the evaluator's seed. 
        # The task says "minimizing all 3 objectives...". Usually, deterministic is safer unless exploration is needed. 
        # However, the baseline used np.random.default_rng(0). I will stick to a local RNG for stability.
        rng = np.random.default_rng(0)

        # Population size
        N = 20
        used = 0
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used += N
        
        # Helper: Fast Non-dominated Sort
        def fast_non_dominated_sort(F):
            n = F.shape[0]
            S = [set() for _ in range(n)]  # S[i] = set of indices j dominated by i
            n_doms = np.zeros(n, dtype=int) # n_doms[i] = number of individuals dominating i
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_doms[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n_doms[i] += 1
            
            front = 0
            current_front = [i for i in range(n) if n_doms[i] == 0]
            for i in current_front:
                rank[i] = front
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            rank[j] = front + 1
                            next_front.append(j)
                front += 1
                current_front = next_front
            return rank

        # Helper: Crowding Distance
        def crowding_distance(F, ranks):
            n = F.shape[0]
            cd = np.zeros(n)
            max_rank = np.max(ranks)
            
            for r in range(max_rank + 1):
                front_indices = np.where(ranks == r)[0]
                if len(front_indices) <= 2:
                    cd[front_indices] = float('inf')
                    continue
                
                for k in range(self.n_obj):
                    f_k = F[front_indices, k]
                    min_f = np.min(f_k)
                    max_f = np.max(f_k)
                    if max_f == min_f:
                        cd[front_indices] = float('inf')
                        continue
                    
                    # Sort by objective k
                    sorted_idx = np.argsort(f_k)
                    sorted_front = front_indices[sorted_idx]
                    cd[sorted_front[0]] = float('inf')
                    cd[sorted_front[-1]] = float('inf')
                    
                    for i in range(1, len(sorted_front) - 1):
                        cd[sorted_front[i]] += (f_k[sorted_idx[i+1]] - f_k[sorted_idx[i-1]]) / (max_f - min_f)
            return cd

        # Helper: Tournament Selection
        def tournament_select(rank, cd, size=2):
            idxs = rng.integers(0, len(rank), size=size)
            best_idx = idxs[0]
            for i in idxs[1:]:
                if rank[i] < rank[best_idx] or (rank[i] == rank[best_idx] and cd[i] > cd[best_idx]):
                    best_idx = i
            return best_idx

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta=20.0):
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(self.dim):
                if rng.random() < 0.5:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        u = rng.random()
                        if u <= 0.5:
                            alpha = (2 * u) ** (1.0 / (eta + 1.0))
                        else:
                            alpha = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                        c1 = 0.5 * ((1 + alpha) * x1[i] + (1 - alpha) * x2[i])
                        c2 = 0.5 * ((1 - alpha) * x1[i] + (1 + alpha) * x2[i])
                        child1[i] = c1
                        child2[i] = c2
            return child1, child2

        # Helper: Polynomial Mutation
        def poly_mutate(x, eta=20.0):
            x_mut = x.copy()
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    delta = (hi[i] - lo[i])
                    if delta < 1e-14:
                        continue
                    u = rng.random()
                    if u < 0.5:
                        delta1 = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta1 = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    x_mut[i] = x[i] + delta1 * delta
            x_mut = np.clip(x_mut, lo, hi)
            return x_mut

        # Main Loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring = []
            for _ in range(N):
                p1_idx = tournament_select(rank, cd)
                p2_idx = tournament_select(rank, cd)
                
                # Crossover
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                
                # Mutation
                c1 = poly_mutate(c1)
                c2 = poly_mutate(c2)
                
                offspring.append(c1)
                if len(offspring) < N: # We need exactly N offspring
                    offspring.append(c2)
                else:
                    # If we generated more, we just take the first N. 
                    # Actually, standard NSGA-II generates N offspring from N pairs.
                    # My loop above generates 2 children per parent pair. 
                    # To get exactly N offspring, I should do N/2 parent pairs? 
                    # Or just generate 1 child per iteration?
                    # Let's simplify: Generate N children.
                    pass
            
            # The above offspring generation is slightly messy. Let's rewrite cleanly.
            offspring_X = []
            for _ in range(N):
                p1_idx = tournament_select(rank, cd)
                p2_idx = tournament_select(rank, cd)
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = poly_mutate(c1)
                # We only need N offspring total. Taking c1 is fine.
                offspring_X.append(c1)
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += N
            
            # Combine
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Sort
            ranks = fast_non_dominated_sort(F_comb)
            cds = crowding_distance(F_comb, ranks)
            
            # Select next generation
            # Sort indices by rank, then by crowding distance (descending)
            order = np.lexsort((cds[::-1], ranks))
            X = X_comb[order[:N]]
            F = F_comb[order[:N]]
            
            # Update ranks and cd for next iteration's selection
            rank = fast_non_dominated_sort(F)
            cd = crowding_distance(F, rank)

        return X
# EVOLVE-BLOCK-END
