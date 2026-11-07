import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size
        # Budget 200. 
        # Init: N
        # Gen 1: N children
        # Gen 2: N children
        # Gen 3: N children
        # Total 4N <= 200 => N <= 50.
        N = 50
        
        # 1. Initialization
        # Split init: 50% random, 50% "informed" for ZDT6
        # ZDT6: f1 depends on x1..x9, g on sum.
        # To get good front, x1..x9 should be small (near 0) to minimize f1.
        # x10 controls the trade-off.
        
        X_init = np.zeros((N, self.dim))
        
        # First half: Random
        X_init[:N//2] = lo + range_ * rng.random((N//2, self.dim))
        
        # Second half: Informed
        # x1..x9 near 0
        x_informed = np.zeros((N - N//2, self.dim))
        # Sample x1..x9 from normal(0, 0.1) clipped to [0,1]
        x_informed[:, :self.dim-1] = np.clip(rng.normal(0, 0.1, (N - N//2, self.dim-1)), 0, 1)
        # Sample x10 uniformly to cover the front
        x_informed[:, self.dim-1] = rng.random(N - N//2)
        X_init[N//2:] = x_informed
        
        F_init = np.array([problem(x) for x in X_init])
        used = N
        
        # Helper: Non-dominated sort and crowding distance
        def non_dominated_sort(F):
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            fronts = []
            S = [[] for _ in range(n)]  # Dominance sets
            n_dominates = np.zeros(n, dtype=int) # Number of points dominated by i
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominates[i] += 1
                        
            current_front = [i for i in range(n) if n_dominates[i] == 0]
            current_rank = 0
            while current_front:
                fronts.append(current_front)
                for i in current_front:
                    ranks[i] = current_rank
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                current_front = next_front
                current_rank += 1
            return ranks, fronts

        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            m = self.n_obj
            for k in range(m):
                f_k = F[front_indices, k]
                min_k = np.min(f_k)
                max_k = np.max(f_k)
                if max_k - min_k < 1e-10:
                    continue
                # Sort indices by f_k
                sorted_idx = np.argsort(f_k)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for i in range(1, n-1):
                    dist[sorted_idx[i]] += (f_k[sorted_idx[i+1]] - f_k[sorted_idx[i-1]]) / (max_k - min_k)
            return dist

        # Selection function: Tournament
        def tournament_select(X, F, k=2):
            n = len(X)
            ranks, fronts = non_dominated_sort(F)
            # Calculate crowding distances for all points? 
            # Efficient way: Calculate for each front, or just use rank + random if fronts are small.
            # For N=50, calculating CD for all is fine.
            cds = np.zeros(n)
            for front in fronts:
                cd_front = crowding_distance(F, front)
                for i, idx in enumerate(front):
                    cds[idx] = cd_front[i]
            
            # Tournament
            winners = []
            for _ in range(2):
                candidates = rng.integers(0, n, size=k)
                best_idx = candidates[0]
                best_rank = ranks[best_idx]
                best_cd = cds[best_idx]
                for c in candidates[1:]:
                    if ranks[c] < best_rank:
                        best_idx = c
                        best_rank = ranks[c]
                        best_cd = cds[c]
                    elif ranks[c] == best_rank and cds[c] > best_cd:
                        best_idx = c
                        best_cd = cds[c]
                winners.append(best_idx)
            return X[winners[0]], X[winners[1]]

        def sbx_crossover(x1, x2, eta=20.0):
            if rng.random() < 0.5:
                return x1.copy(), x2.copy()
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for i in range(self.dim):
                if x1[i] == x2[i]:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
                else:
                    if x1[i] < x2[i]:
                        lo_i, hi_i = x1[i], x2[i]
                    else:
                        lo_i, hi_i = x2[i], x1[i]
                    
                    # SBX logic
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1.0))
                    
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return child1, child2

        def polynomial_mutation(x, eta=20.0):
            # Simplified Gaussian mutation for speed and reliability in small budget
            # Or standard polynomial
            # Let's use Gaussian with adaptive step
            sigma = 0.1
            mut = x + sigma * range_ * rng.standard_normal(self.dim)
            return np.clip(mut, lo, hi)

        # Main Loop
        X = X_init
        F = F_init
        
        # We have 150 budget left.
        # 3 generations of 50 children each.
        for gen in range(3):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
                
            children_X = np.zeros((N, self.dim))
            children_F = np.zeros((N, self.n_obj))
            
            for i in range(N):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                p1, p2 = tournament_select(X, F)
                
                # 10% chance of random re-init to escape local minima
                if rng.random() < 0.1:
                    c1 = lo + range_ * rng.random(self.dim)
                    c2 = lo + range_ * rng.random(self.dim)
                else:
                    c1, c2 = sbx_crossover(p1, p2)
                    c1 = polynomial_mutation(c1)
                    c2 = polynomial_mutation(c2)
                
                f1 = problem(c1)
                used += 1
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                f2 = problem(c2)
                used += 1
                
                children_X[i] = c1
                children_F[i] = f1
                # If budget allows, store second child in next slot? 
                # No, we fixed N children per gen. 
                # We can just use the first child to save budget or evaluate both if budget permits.
                # With N=50, 2 children per parent = 100 children. 
                # Budget 150 left. 
                # Gen 1: 100 evals? No, 200 total.
                # Init 50.
                # If we do 2 children per parent, we use 2*N evals per gen.
                # 2*50 = 100.
                # Gen 1: 100 evals. Total 150.
                # Gen 2: 50 evals left. Can only do 25 parents (50 children).
                
                # Let's adjust strategy:
                # Init: 50
                # Gen 1: 100 children (50 parents x 2). Total 150.
                # Gen 2: 50 children (25 parents x 2). Total 200.
                
                # This is complex to loop dynamically. 
                # Simpler: Loop while budget > 0.
                # But we need to structure the population update.
                
                # Let's restart the loop logic to be simpler:
                # Just keep producing children until budget is done.
                # But we need to update X and F.
                
                # Let's break out and use a while loop instead of for gen.
                pass 

        # The previous loop structure was flawed for 2 children.
        # Let's rewrite the main loop properly.
        
        X = X_init
        F = F_init
        
        while used < self.budget and time.time() < deadline:
            # Produce 2 children
            p1, p2 = tournament_select(X, F)
            
            if rng.random() < 0.1:
                c1 = lo + range_ * rng.random(self.dim)
                c2 = lo + range_ * rng.random(self.dim)
            else:
                c1, c2 = sbx_crossover(p1, p2)
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
            
            f1 = problem(c1)
            used += 1
            if used >= self.budget:
                break
            f2 = problem(c2)
            used += 1
            
            # Combine and Select
            # Current pop: X (N), F (N)
            # Children: c1, c2
            # We want to keep best N from X + [c1, c2] ?
            # No, that's (N+2) -> N. 
            # This is steady-state.
            # Or we can accumulate children and do generational replacement?
            # With small budget, steady state is often better for convergence.
            
            # Let's do generational replacement every 25 children (12.5 parents)?
            # No, let's just do:
            # Add children to population, if size > N, select best N.
            
            X_new = np.vstack([X, c1, c2])
            F_new = np.vstack([F, f1, f2])
            
            # Select best N from X_new
            ranks, fronts = non_dominated_sort(F_new)
            # We need to pick N points.
            # Priority: Lower rank, higher crowding.
            
            # Sort by rank, then by CD (descending)
            # Calculate CD for all?
            cds = np.zeros(len(F_new))
            for front in fronts:
                cd_front = crowding_distance(F_new, front)
                for i, idx in enumerate(front):
                    cds[idx] = cd_front[i]
            
            # Sort indices: Rank asc, CD desc
            sort_keys = np.lexsort((cds * -1, ranks))
            # Take first N
            best_indices = sort_keys[:N]
            
            X = X_new[best_indices]
            F = F_new[best_indices]
            
            # Ensure we don't exceed budget in next iteration
            if used >= self.budget:
                break

        return X
