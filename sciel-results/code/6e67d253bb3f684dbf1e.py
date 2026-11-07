"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """
    Multi-objective optimizer using NSGA-II.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine population size and generations
        # We want to use exactly the budget.
        # Init: N_init, Offspring: N_gen * N_pop
        # Let N_pop = 20.
        # N_init = 20.
        # Remaining = 180.
        # N_gen = 180 // 20 = 9.
        
        pop_size = 20
        if self.budget < 2 * pop_size:
             pop_size = self.budget // 2
             if pop_size < 2:
                 pop_size = 2
                 # Fallback to simple random search if budget is tiny
                 X = np.random.rand(self.budget, self.dim)
                 X = lo + (hi - lo) * X
                 return np.array([problem(x) for x in X])
        
        n_init = pop_size
        n_gen = (self.budget - n_init) // pop_size
        
        # 1. Initialization
        X = lo + (hi - lo) * np.random.rand(n_init, self.dim)
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            fronts = []
            S = [[] for _ in range(n)] # S[i] = indices j dominated by i
            n_dominates = np.zeros(n, dtype=int) # n_dominates[i] = count of i dominating j
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominates[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominates[i] += 1
            
            current_front = [i for i in range(n) if n_dominates[i] == 0]
            rank = 0
            while current_front:
                fronts.append(current_front)
                for i in current_front:
                    ranks[i] = rank
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                current_front = next_front
                rank += 1
            return fronts, ranks

        # Helper: Crowding distance
        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n == 0:
                return {}
            if n == 1:
                return {front_indices[0]: float('inf')}
            
            dist = np.zeros(n)
            F_front = F[front_indices]
            for m in range(self.n_obj):
                min_val = np.min(F_front[:, m])
                max_val = np.max(F_front[:, m])
                if max_val == min_val:
                    continue
                # Sort indices by objective m
                sorted_idx = np.argsort(F_front[:, m])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                for k in range(1, n - 1):
                    i = sorted_idx[k]
                    i_prev = sorted_idx[k - 1]
                    i_next = sorted_idx[k + 1]
                    dist[i] += (F_front[i_next, m] - F_front[i_prev, m]) / (max_val - min_val)
            
            cd = {}
            for i, idx in enumerate(front_indices):
                cd[idx] = dist[i]
            return cd

        # Helper: Selection (Tournament)
        def tournament_select(fronts, ranks, cd_cache):
            # cd_cache maps rank -> {idx: dist}
            # We need a unified crowding distance for comparison
            # Standard NSGA-II: lower rank is better. If same rank, higher crowding distance is better.
            
            def better(i, j):
                if ranks[i] != ranks[j]:
                    return ranks[i] < ranks[j]
                # Get crowding distance
                # We need to ensure cd is calculated for the front containing i and j
                # In standard NSGA-II, cd is calculated for fronts as they are assigned.
                # Here we pre-calculate for all fronts in the combined population.
                cd_i = cd_cache.get(ranks[i], {}).get(i, 0)
                cd_j = cd_cache.get(ranks[j], {}).get(j, 0)
                return cd_i > cd_j

            # Select 2 random parents
            i = np.random.randint(len(ranks))
            j = np.random.randint(len(ranks))
            if better(i, j):
                return i
            else:
                return j

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2):
            if np.random.rand() < 0.9: # Crossover probability
                eta_c = 20.0
                x1, x2 = x1.copy(), x2.copy()
                for k in range(self.dim):
                    if x1[k] != x2[k]:
                        if np.random.rand() < 0.5:
                            x1[k], x2[k] = x2[k], x1[k]
                        
                        if x1[k] < x2[k]:
                            x1[k], x2[k] = x2[k], x1[k]
                            # Now x1 >= x2
                            if x1[k] - x2[k] < 1e-14:
                                continue
                            
                            u = np.random.rand()
                            beta = 1.0 + 2.0 * (x1[k] - x2[k])
                            alpha = 2.0 - beta ** (eta_c + 1.0)
                            if u <= 1.0 / alpha:
                                beta_q = (u * alpha) ** (1.0 / (eta_c + 1.0))
                            else:
                                beta_q = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))
                            
                            c1 = 0.5 * (x1[k] + x2[k] - beta_q * (x1[k] - x2[k]))
                            c2 = 0.5 * (x1[k] + x2[k] + beta_q * (x1[k] - x2[k]))
                            
                            c1 = np.clip(c1, lo[k], hi[k])
                            c2 = np.clip(c2, lo[k], hi[k])
                            
                            # Swap back if we swapped
                            if np.random.rand() < 0.5:
                                x1[k], x2[k] = c2, c1
                            else:
                                x1[k], x2[k] = c1, c2
            return x1, x2

        # Helper: Polynomial Mutation
        def poly_mutation(x):
            if np.random.rand() < 1.0 / self.dim: # Mutation probability per gene
                eta_m = 20.0
                x = x.copy()
                for k in range(self.dim):
                    if np.random.rand() < 1.0 / self.dim:
                        u = np.random.rand()
                        delta1 = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                        delta2 = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                        delta = delta1 if np.random.rand() < 0.5 else delta2
                        x[k] += delta * (hi[k] - lo[k])
                        x[k] = np.clip(x[k], lo[k], hi[k])
            return x

        # Main Loop
        for gen in range(n_gen):
            # Create offspring
            X_offspring = np.zeros((pop_size, self.dim))
            for i in range(pop_size):
                # Select parents
                p1 = tournament_select(fronts, ranks, cd_cache)
                p2 = tournament_select(fronts, ranks, cd_cache)
                
                # Crossover
                c1, c2 = sbx_crossover(X[p1], X[p2])
                
                # Mutation
                c1 = poly_mutation(c1)
                c2 = poly_mutation(c2)
                
                # We need to fill pop_size offspring. 
                # This simple loop produces 2 per parent pair, but we only need pop_size.
                # Standard NSGA-II: generate pop_size/2 pairs? Or pop_size individuals?
                # Let's generate pop_size individuals by picking 2 parents and doing crossover/mutation.
                # To keep it simple and strictly within budget, we generate 1 offspring per iteration of this loop?
                # No, SBX produces 2. 
                # Let's adjust: Generate pop_size // 2 pairs, yielding pop_size offspring (if pop_size is even).
                # If pop_size is odd, we might have issues. Let's assume pop_size is even (20).
                
                # Actually, the loop above runs pop_size times, producing 2*pop_size individuals.
                # We only need pop_size.
                # Let's restructure:
                pass
            
            # Restructure offspring generation to exactly pop_size
            X_offspring = np.zeros((pop_size, self.dim))
            for i in range(pop_size // 2):
                p1 = tournament_select(fronts, ranks, cd_cache)
                p2 = tournament_select(fronts, ranks, cd_cache)
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = poly_mutation(c1)
                c2 = poly_mutation(c2)
                X_offspring[2*i] = c1
                X_offspring[2*i+1] = c2
            if pop_size % 2 != 0:
                # Handle odd case by cloning last one or random
                X_offspring[-1] = X[np.random.randint(pop_size)]

            F_offspring = np.array([problem(x) for x in X_offspring])
            used += pop_size
            
            # Combine
            X_combined = np.vstack([X, X_offspring])
            F_combined = np.vstack([F, F_offspring])
            
            # Non-dominated sort
            fronts, ranks = non_dominated_sort(F_combined)
            
            # Calculate crowding distance for each front
            cd_cache = {}
            for front in fronts:
                cd_cache[ranks[front[0]]] = crowding_distance(F_combined, front)
            
            # Select next population
            X_next = np.zeros((pop_size, self.dim))
            F_next = np.zeros((pop_size, self.n_obj))
            selected_indices = []
            
            # Fill by rank
            for front in fronts:
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    # Need to pick from this front based on crowding distance
                    needed = pop_size - len(selected_indices)
                    front_cd = cd_cache[ranks[front[0]]]
                    sorted_front = sorted(front, key=lambda idx: front_cd[idx], reverse=True)
                    selected_indices.extend(sorted_front[:needed])
                    break
            
            X = X_combined[selected_indices]
            F = F_combined[selected_indices]
            
            # Update fronts and ranks for next generation selection?
            # No, we re-sort in the next iteration.
            # But we need fronts and ranks for the tournament selection in the NEXT generation's offspring creation.
            # So we must recalculate fronts/ranks for the NEW population X, F.
            fronts, ranks = non_dominated_sort(F)
            cd_cache = {}
            for front in fronts:
                cd_cache[ranks[front[0]]] = crowding_distance(F, front)

        return F
# EVOLVE-BLOCK-END
