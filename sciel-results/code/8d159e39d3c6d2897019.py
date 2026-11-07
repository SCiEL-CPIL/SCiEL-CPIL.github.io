# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for RE37 using NSGA-II.
    
    NSGA-II is a widely used evolutionary algorithm for multi-objective optimization.
    It uses non-dominated sorting to rank solutions and crowding distance to maintain
    diversity, which is effective for maximizing hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Track all evaluations
        all_X = []
        all_F = []
        used = 0
        
        def evaluate(x):
            nonlocal used
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(problem(x), dtype=float)
            all_X.append(x.copy())
            all_F.append(f.copy())
            used += 1
            return f
        
        # Initialize population
        # We need a population size that fits within the budget.
        # Let's use a population size of 20.
        pop_size = 20
        # Ensure we have enough budget for at least one generation
        if self.budget < pop_size * 2:
             pop_size = max(2, self.budget // 2)
             
        # Initial random population
        population = np.random.uniform(lo, hi, (pop_size, self.dim))
        # Evaluate initial population
        F = np.array([evaluate(x) for x in population])
        
        # Helper functions for NSGA-II
        
        def fast_non_dominated_sort(F):
            """
            Returns ranks (list of lists of indices) for each point in F.
            F is an array of shape (N, M).
            """
            N = len(F)
            M = F.shape[1]
            S = [[] for _ in range(N)]
            n = np.zeros(N, dtype=int)
            rank = np.zeros(N, dtype=int)
            
            for p in range(N):
                for q in range(N):
                    if p == q:
                        continue
                    # Check if p dominates q
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    # Check if q dominates p
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n[p] += 1
            
            front = [0]
            i = 0
            while front:
                next_front = []
                for p in front:
                    rank[p] = i
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            next_front.append(q)
                front = next_front
                i += 1
            
            # Convert to list of lists
            ranks = [[] for _ in range(i)]
            for p in range(N):
                ranks[rank[p]].append(p)
            return ranks

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for points with given indices.
            Returns array of distances for those indices.
            """
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            N = len(indices)
            M = F.shape[1]
            dist = np.zeros(N)
            F_sub = F[indices]
            
            for m in range(M):
                # Sort indices by m-th objective
                order = np.argsort(F_sub[:, m])
                # Map sorted order back to original indices in indices list
                # Actually, we just need to assign distance based on position in sorted order
                # dist[order[0]] and dist[order[-1]] are inf
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                
                min_val = F_sub[order[0], m]
                max_val = F_sub[order[-1], m]
                if max_val == min_val:
                    continue
                    
                for k in range(1, N - 1):
                    dist[order[k]] += (F_sub[order[k+1], m] - F_sub[order[k-1], m]) / (max_val - min_val)
            
            return dist

        def tournament_selection(F, ranks, crowd, k=2):
            """
            Select one individual using tournament selection.
            ranks: list of lists of indices.
            crowd: dict or array mapping index to crowding distance.
            """
            idxs = np.random.choice(len(F), size=k, replace=False)
            best_idx = idxs[0]
            best_rank = ranks.index([i for i in ranks if i == best_idx][0]) if best_idx in [i for i in ranks[0]] else 0
            # Efficient rank lookup: create a rank array
            # We'll pass a rank_array instead of ranks list for efficiency in loop
            
            # Let's refactor: pass rank_array and crowd_array
            # This function is a bit messy with the current structure. 
            # Let's implement selection inside the main loop where we have arrays.
            pass # Placeholder, will implement logic in main loop

        # Main NSGA-II Loop
        # We will run until budget is exhausted.
        # Each generation involves:
        # 1. Creating offspring (pop_size evaluations)
        # 2. Combining parent and offspring (2 * pop_size points)
        # 3. Sorting and selecting next population (no evaluations)
        
        # To fit budget, we determine how many generations we can afford.
        # Initial evals: pop_size.
        # Each gen: pop_size new evals.
        # Total evals = pop_size + G * pop_size <= budget
        # G = (budget - pop_size) // pop_size
        
        # However, we can also just run a loop and break when used >= budget.
        
        # Current population indices in all_X
        # all_X contains all history. We need to track which ones are current.
        # It's easier to keep current population as indices into all_X/all_F or just as arrays.
        # Let's keep current population as arrays P_X and P_F, and their indices in global history if needed.
        # But since we only care about the final set of all evaluated points for hypervolume,
        # we just need to ensure we evaluate new points.
        
        # Let's manage population as arrays.
        P_X = population.copy()
        P_F = F.copy()
        
        # Precompute ranks and crowding for current population
        ranks = fast_non_dominated_sort(P_F)
        # Calculate crowding distance for each front
        crowd = np.zeros(len(P_F))
        for front in ranks:
            if len(front) > 0:
                d = crowding_distance(P_F, front)
                for i, idx in enumerate(front):
                    crowd[idx] = d[i]
        
        # Selection function using rank and crowd
        def select_one(P_F, ranks_list, crowd_arr):
            # Create rank array for quick lookup
            rank_arr = np.zeros(len(P_F), dtype=int)
            for r, front in enumerate(ranks_list):
                for idx in front:
                    rank_arr[idx] = r
            
            idxs = np.random.choice(len(P_F), size=2, replace=False)
            i1, i2 = idxs
            if rank_arr[i1] < rank_arr[i2]:
                return i1
            elif rank_arr[i2] < rank_arr[i1]:
                return i2
            else:
                # Same rank, compare crowding distance
                if crowd_arr[i1] > crowd_arr[i2]:
                    return i1
                else:
                    return i2

        # Crossover and Mutation
        def sbx_crossover(x1, x2, eta=20.0):
            if rng.random() < 0.9: # Crossover probability
                c1 = np.empty_like(x1)
                c2 = np.empty_like(x2)
                for i in range(len(x1)):
                    if rng.random() < 0.5:
                        # Generate beta
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1.0))
                        else:
                            beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                        
                        c1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                        c2[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
                    else:
                        c1[i] = x1[i]
                        c2[i] = x2[i]
                # Polynomial mutation
                for i in range(len(c1)):
                    if rng.random() < 1.0 / len(c1):
                        u = rng.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                        else:
                            delta = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                        c1[i] += delta * (hi[i] - lo[i])
                        c2[i] += delta * (hi[i] - lo[i])
                
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                return c1, c2
            else:
                return x1.copy(), x2.copy()

        # Run generations
        while used + pop_size <= self.budget:
            # Create offspring
            offspring_X = []
            for _ in range(pop_size // 2):
                if used >= self.budget:
                    break
                p1_idx = select_one(P_F, ranks, crowd)
                p2_idx = select_one(P_F, ranks, crowd)
                c1, c2 = sbx_crossover(P_X[p1_idx], P_X[p2_idx])
                offspring_X.append(c1)
                offspring_X.append(c2)
            
            if not offspring_X:
                break
                
            # Evaluate offspring
            offspring_F = np.array([evaluate(x) for x in offspring_X])
            
            # Combine parent and offspring
            combined_X = np.vstack([P_X, np.array(offspring_X)])
            combined_F = np.vstack([P_F, offspring_F])
            
            # Sort combined population
            combined_ranks = fast_non_dominated_sort(combined_F)
            combined_crowd = np.zeros(len(combined_F))
            for front in combined_ranks:
                if len(front) > 0:
                    d = crowding_distance(combined_F, front)
                    for i, idx in enumerate(front):
                        combined_crowd[idx] = d[i]
            
            # Select next population
            new_P_X = []
            new_P_F = []
            selected_indices = set()
            
            for front in combined_ranks:
                if len(new_P_X) + len(front) <= pop_size:
                    for idx in front:
                        new_P_X.append(combined_X[idx])
                        new_P_F.append(combined_F[idx])
                        selected_indices.add(idx)
                else:
                    # Need to select from this front based on crowding distance
                    need = pop_size - len(new_P_X)
                    # Get indices in this front
                    front_indices = np.array(front)
                    front_crowd = combined_crowd[front_indices]
                    # Sort by crowding distance descending
                    order = np.argsort(front_crowd)[::-1]
                    for i in range(need):
                        idx = front_indices[order[i]]
                        new_P_X.append(combined_X[idx])
                        new_P_F.append(combined_F[idx])
                        selected_indices.add(idx)
                    break
            
            P_X = np.array(new_P_X)
            P_F = np.array(new_P_F)
            
            # Update ranks and crowding for next generation
            ranks = fast_non_dominated_sort(P_F)
            crowd = np.zeros(len(P_F))
            for front in ranks:
                if len(front) > 0:
                    d = crowding_distance(P_F, front)
                    for i, idx in enumerate(front):
                        crowd[idx] = d[i]

        # Return last X (ignored by evaluator, but keep interface)
        if all_X:
            return all_X[-1]
        else:
            return lo.copy()
# EVOLVE-BLOCK-END
