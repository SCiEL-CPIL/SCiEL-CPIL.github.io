import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Budget allocation
        # 200 total. 
        # Let's use a population size of 50.
        # Initial random: 50
        # Remaining: 150.
        # Generations: 3 generations of 50 offspring? Or continuous replacement?
        # Let's do a proper NSGA-II style but with fixed budget.
        # Since budget is small (200), a small population with many generations or a larger population with few?
        # With 4 variables, 50 is reasonable.
        
        N = 50
        
        # 1. Initial Population
        # Use global np.random to respect seed
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper: Non-dominated sort (vectorized where possible, but N is small so loops are fine)
        def non_dominated_sort(F):
            M = len(F)
            ranks = np.zeros(M, dtype=int)
            fronts = []
            S = [list() for _ in range(M)]
            n = np.zeros(M, dtype=int)
            
            for p in range(M):
                for q in range(p + 1, M):
                    # Check p dominates q
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                        n[q] += 1
                    # Check q dominates p
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        S[q].append(p)
                        n[p] += 1
            
            k = 0
            current_front = [p for p in range(M) if n[p] == 0]
            for p in current_front:
                ranks[p] = 0
            fronts.append(current_front)
            
            while fronts[k]:
                next_front = []
                for p in fronts[k]:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            ranks[q] = k + 1
                            next_front.append(q)
                if next_front:
                    k += 1
                    fronts.append(next_front)
                else:
                    break
            
            return ranks, fronts

        # Helper: Crowding distance
        def crowding_distance(F, front_indices):
            cd = {i: 0.0 for i in front_indices}
            if len(front_indices) <= 2:
                for i in front_indices:
                    cd[i] = float('inf')
                return cd
            
            m = self.n_obj
            for k in range(m):
                sorted_indices = sorted(front_indices, key=lambda i: F[i][k])
                cd[sorted_indices[0]] = float('inf')
                cd[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0]][k]
                f_max = F[sorted_indices[-1]][k]
                if f_max == f_min:
                    continue
                for i in range(1, len(sorted_indices) - 1):
                    i_curr = sorted_indices[i]
                    i_next = sorted_indices[i+1]
                    i_prev = sorted_indices[i-1]
                    cd[i_curr] += (F[i_next][k] - F[i_prev][k]) / (f_max - f_min)
            return cd

        # Helper: Tournament Selection
        def tournament_select(F, ranks, cd_dict, size=3):
            candidates = np.random.choice(len(F), size, replace=False)
            best = candidates[0]
            for c in candidates[1:]:
                if ranks[c] < ranks[best]:
                    best = c
                elif ranks[c] == ranks[best]:
                    if cd_dict.get(c, 0) > cd_dict.get(best, 0):
                        best = c
            return best

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2):
            x1 = np.array(x1); x2 = np.array(x2)
            child1 = x1.copy(); child2 = x2.copy()
            eta = 20.0
            for j in range(self.dim):
                if np.random.random() < 0.5:
                    if abs(x1[j] - x2[j]) > 1e-14:
                        u = np.random.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (2 * (1 - u)) ** (-1 / (eta + 1))
                        
                        child1[j] = 0.5 * ((1 + beta) * x1[j] + (1 - beta) * x2[j])
                        child2[j] = 0.5 * ((1 - beta) * x1[j] + (1 + beta) * x2[j])
            # Clip
            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            return child1, child2

        # Helper: Polynomial Mutation
        def polynomial_mutation(x):
            x = np.array(x, dtype=float)
            eta = 20.0
            for j in range(self.dim):
                if np.random.random() < 1.0 / self.dim:
                    u = np.random.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    
                    x[j] += delta_q * (hi[j] - lo[j])
            return np.clip(x, lo, hi)

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Compute ranks and crowding
            ranks, fronts = non_dominated_sort(F)
            cd_dict = {}
            for front in fronts:
                if not front:
                    break
                cd_dict.update(crowding_distance(F, front))
            
            # Select parents
            p1 = tournament_select(F, ranks, cd_dict)
            p2 = tournament_select(F, ranks, cd_dict)
            
            # Generate offspring
            c1, c2 = sbx_crossover(X[p1], X[p2])
            c1 = polynomial_mutation(c1)
            c2 = polynomial_mutation(c2)
            
            # Evaluate and replace worst
            # We will replace one by one to strictly respect budget
            f1 = problem(c1)
            used += 1
            
            # Find worst individual in current population
            # Worst: highest rank, then lowest crowding
            worst_idx = 0
            worst_rank = -1
            worst_cd = float('inf')
            for i in range(N):
                if ranks[i] > worst_rank:
                    worst_rank = ranks[i]
                    worst_idx = i
                    worst_cd = cd_dict.get(i, 0)
                elif ranks[i] == worst_rank:
                    if cd_dict.get(i, 0) < worst_cd:
                        worst_cd = cd_dict.get(i, 0)
                        worst_idx = i
                        
            X[worst_idx] = c1
            F[worst_idx] = f1
            
            if used < self.budget:
                f2 = problem(c2)
                used += 1
                
                # Re-rank to find new worst for second offspring
                ranks, fronts = non_dominated_sort(F)
                cd_dict = {}
                for front in fronts:
                    if not front:
                        break
                    cd_dict.update(crowding_distance(F, front))
                    
                worst_idx2 = 0
                worst_rank2 = -1
                worst_cd2 = float('inf')
                for i in range(N):
                    if ranks[i] > worst_rank2:
                        worst_rank2 = ranks[i]
                        worst_idx2 = i
                        worst_cd2 = cd_dict.get(i, 0)
                    elif ranks[i] == worst_rank2:
                        if cd_dict.get(i, 0) < worst_cd2:
                            worst_cd2 = cd_dict.get(i, 0)
                            worst_idx2 = i
                            
                X[worst_idx2] = c2
                F[worst_idx2] = f2

        return X
