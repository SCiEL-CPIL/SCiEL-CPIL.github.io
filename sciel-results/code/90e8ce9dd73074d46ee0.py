import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        # Fixed population size suitable for small budgets
        # We want to allow a few generations. 
        # If pop_size is N, each generation uses N evaluations.
        # Total evaluations = N (init) + (budget - N) / N * N approx budget.
        # Let's pick a pop size that divides budget reasonably or is small enough for many iterations.
        # For 200 budget, pop size 20 gives ~10 generations.
        # Pop size 10 gives ~20 generations.
        # Let's use 20 for better diversity per generation.
        self.pop_size = 20 

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        pop_size = self.pop_size
        
        # Ensure pop_size is at least 2 and does not exceed budget
        if pop_size > budget:
            pop_size = max(2, budget // 2)
            
        # Initialize population
        X = lo + (hi - lo) * np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II parameters
        # Crossover probability
        pc = 0.9
        # Distribution index for SBX and PM
        eta_c = 15.0
        eta_m = 20.0
        
        while used < budget:
            if time.time() > deadline:
                break
                
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            # We can generate up to pop_size offspring per generation
            # But we must respect the remaining budget
            remaining = budget - used
            num_offspring = min(pop_size, remaining)
            
            if num_offspring <= 0:
                break

            for _ in range(num_offspring):
                if time.time() > deadline:
                    break
                    
                # Selection: Binary Tournament
                i1, i2 = np.random.choice(pop_size, 2, replace=False)
                # Simple dominance check for selection
                # If i1 dominates i2, pick i1. If i2 dominates i1, pick i2.
                # Else, pick randomly (could use crowding distance, but simple is fine for small pop)
                if np.all(F[i1] <= F[i2]) and np.any(F[i1] < F[i2]):
                    p1 = X[i1]
                elif np.all(F[i2] <= F[i1]) and np.any(F[i2] < F[i1]):
                    p1 = X[i2]
                else:
                    p1 = X[i1] if np.random.rand() < 0.5 else X[i2]
                
                i3, i4 = np.random.choice(pop_size, 2, replace=False)
                if np.all(F[i3] <= F[i4]) and np.any(F[i3] < F[i4]):
                    p2 = X[i3]
                elif np.all(F[i4] <= F[i3]) and np.any(F[i4] < F[i3]):
                    p2 = X[i4]
                else:
                    p2 = X[i3] if np.random.rand() < 0.5 else X[i4]
                
                # Crossover: SBX
                child = np.copy(p1)
                if np.random.rand() < pc:
                    for j in range(dim):
                        if np.random.rand() < 0.5:
                            # SBX formula
                            u = np.random.rand()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta_c + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                            
                            # Standard SBX
                            c1 = 0.5 * ((1 + beta) * p1[j] + (1 - beta) * p2[j])
                            c2 = 0.5 * ((1 - beta) * p1[j] + (1 + beta) * p2[j])
                            
                            # Clip to bounds
                            c1 = np.clip(c1, lo[j], hi[j])
                            c2 = np.clip(c2, lo[j], hi[j])
                            
                            # We only produce one child per pair in this loop to keep logic simple
                            # But usually SBX produces 2. Let's just take one.
                            child[j] = c1 if np.random.rand() < 0.5 else c2
                
                # Mutation: Polynomial Mutation
                for j in range(dim):
                    if np.random.rand() < 1.0 / dim:
                        u = np.random.rand()
                        if u <= 0.5:
                            delta1 = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                        else:
                            delta1 = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                        
                        child[j] += delta1 * (hi[j] - lo[j])
                        child[j] = np.clip(child[j], lo[j], hi[j])
                
                offspring_X.append(child)
                fc = problem(child)
                offspring_F.append(fc)
                used += 1
            
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parents and offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])
            
            # Non-dominated sorting
            ranks = self.non_dominated_sort(F_combined)
            
            # Crowding distance
            crowd = self.crowding_distance(F_combined, ranks)
            
            # Selection: Keep best pop_size individuals
            # Sort by rank (ascending), then by crowding distance (descending)
            # We want to select the top pop_size
            indices = np.arange(len(X_combined))
            
            # Create a key for sorting: (rank, -crowding)
            # We can't just sort tuples easily if we want to keep original indices.
            # Let's sort the combined arrays based on rank and crowd.
            
            # Assign final selection
            selected_indices = []
            for r in range(np.max(ranks) + 1):
                idx_in_rank = np.where(ranks == r)[0]
                if len(idx_in_rank) == 0:
                    continue
                
                # Sort by crowding distance descending
                sorted_idx = idx_in_rank[np.argsort(-crowd[idx_in_rank])]
                
                if len(selected_indices) + len(sorted_idx) <= pop_size:
                    selected_indices.extend(sorted_idx.tolist())
                else:
                    # Take only as many as needed
                    needed = pop_size - len(selected_indices)
                    selected_indices.extend(sorted_idx[:needed].tolist())
                    break
            
            X = X_combined[selected_indices]
            F = F_combined[selected_indices]
            
        return X

    def non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]  # Set of solutions dominated by i
        n_d = [0] * n  # Number of solutions dominating i
        rank = [0] * n
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    n_d[p] += 1
        
        current_front = []
        for p in range(n):
            if n_d[p] == 0:
                rank[p] = 0
                current_front.append(p)
        
        i = 0
        while current_front:
            next_front = []
            for p in current_front:
                for q in S[p]:
                    n_d[q] -= 1
                    if n_d[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            current_front = next_front
            i += 1
        
        return rank

    def crowding_distance(self, F, ranks):
        n = len(F)
        crowd = np.zeros(n)
        
        # Group by rank
        unique_ranks = np.unique(ranks)
        for r in unique_ranks:
            idx = np.where(ranks == r)[0]
            if len(idx) <= 1:
                crowd[idx] = np.inf
                continue
            
            # For each objective, sort and assign distance
            for k in range(F.shape[1]):
                f_vals = F[idx, k]
                min_val = np.min(f_vals)
                max_val = np.max(f_vals)
                
                if max_val - min_val < 1e-12:
                    # All values are the same
                    crowd[idx] += 1.0 # Arbitrary small addition or 0? Usually inf or 0. Let's use 0 to not bias.
                    continue
                
                # Sort indices by objective value
                sorted_idx = idx[np.argsort(f_vals)]
                
                # Assign infinite distance to extremes
                crowd[sorted_idx[0]] = np.inf
                crowd[sorted_idx[-1]] = np.inf
                
                # Assign distance to others
                for i in range(1, len(sorted_idx) - 1):
                    prev_val = f_vals[sorted_idx[i-1]]
                    next_val = f_vals[sorted_idx[i+1]]
                    dist = (next_val - prev_val) / (max_val - min_val)
                    crowd[sorted_idx[i]] += dist
                    
        return crowd
