"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # With budget 200, N=20 allows for 1 initial gen + 9 offspring gens = 200 evals.
        N = 20
        if N > self.budget:
            N = self.budget
            
        # Initialize population using np.random to respect global seed
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper: Fast Non-Dominated Sort
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_doms = [0] * n
            ranks = [0] * n
            front = [set() for _ in range(n)]
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_doms[p] += 1
                if n_doms[p] == 0:
                    ranks[p] = 0
                    front[0].add(p)
                    
            i = 0
            while front[i]:
                next_front = set()
                for p in front[i]:
                    for q in S[p]:
                        n_doms[q] -= 1
                        if n_doms[q] == 0:
                            ranks[q] = i + 1
                            next_front.add(q)
                i += 1
                if next_front:
                    front[i] = next_front
                else:
                    break
            return [list(f) for f in front if f]

        # Helper: Crowding Distance
        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return {i: float('inf') for i in front_indices}
            
            m = len(F)
            n_obj = self.n_obj
            crowd = np.zeros(m)
            
            for k in range(n_obj):
                sorted_idx = sorted(front_indices, key=lambda i: F[i][k])
                crowd[sorted_idx[0]] = float('inf')
                crowd[sorted_idx[-1]] = float('inf')
                f_min = F[sorted_idx[0]][k]
                f_max = F[sorted_idx[-1]][k]
                if f_max == f_min:
                    continue
                for i in range(1, len(sorted_idx) - 1):
                    idx = sorted_idx[i]
                    crowd[idx] += (F[sorted_idx[i+1]][k] - F[sorted_idx[i-1]][k]) / (f_max - f_min)
            
            return {i: crowd[i] for i in front_indices}

        # Helper: Binary Tournament
        def tournament_select(pop_F, ranks, crowds):
            i = np.random.randint(len(pop_F))
            j = np.random.randint(len(pop_F))
            if ranks[i] < ranks[j]:
                return i
            elif ranks[j] < ranks[i]:
                return j
            else:
                if crowds[i] > crowds[j]:
                    return i
                else:
                    return j

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta=20.0):
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    continue
                if abs(x1[i] - x2[i]) < 1e-14:
                    continue
                if x1[i] > x2[i]:
                    x1[i], x2[i] = x2[i], x1[i]
                
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                
                child1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                child2[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
            return child1, child2

        # Helper: Polynomial Mutation
        def polynomial_mutate(x, eta=20.0):
            x_mut = x.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    continue
                u = np.random.rand()
                if u <= 0.5:
                    delta_q = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta_q = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                
                x_mut[i] += delta_q * (hi[i] - lo[i])
            return np.clip(x_mut, lo, hi)

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Sort current population
            fronts = fast_non_dominated_sort(F)
            ranks = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
            
            # Calculate crowding distance for all fronts
            crowds = np.zeros(len(F))
            for front in fronts:
                cd_dict = crowding_distance(F, front)
                for idx, c in cd_dict.items():
                    crowds[idx] = c

            # Generate offspring
            Q_X = []
            Q_F = []
            
            # We can generate up to min(N, self.budget - used) offspring
            num_offspring = min(N, self.budget - used)
            
            for _ in range(num_offspring):
                # Selection
                p1_idx = tournament_select(F, ranks, crowds)
                p2_idx = tournament_select(F, ranks, crowds)
                
                # Crossover
                child1, child2 = sbx_crossover(X[p1_idx], X[p2_idx])
                
                # Mutation
                child1 = polynomial_mutate(child1)
                child2 = polynomial_mutate(child2)
                
                # Evaluate
                f1 = problem(child1)
                f2 = problem(child2)
                
                Q_X.append(child1)
                Q_F.append(f1)
                used += 1
                if used >= self.budget:
                    break
                
                Q_X.append(child2)
                Q_F.append(f2)
                used += 1
                if used >= self.budget:
                    break
            
            if not Q_X:
                break

            Q_X = np.array(Q_X)
            Q_F = np.array(Q_F)
            
            # Combine P and Q
            R_X = np.vstack((X, Q_X))
            R_F = np.vstack((F, Q_F))
            
            # Sort R
            fronts_R = fast_non_dominated_sort(R_F)
            
            # Select N individuals for next generation
            next_X = []
            next_F = []
            next_ranks = np.zeros(len(R_F), dtype=int)
            for i, front in enumerate(fronts_R):
                for idx in front:
                    next_ranks[idx] = i
            
            # We need to select N individuals from R
            # Iterate through fronts, adding entire fronts until we exceed N
            # Then fill the rest from the next front based on crowding distance
            selected_indices = []
            for front in fronts_R:
                if len(selected_indices) + len(front) <= N:
                    selected_indices.extend(front)
                else:
                    # Need to pick only some from this front
                    # Calculate crowding distance for this front
                    cd_dict = crowding_distance(R_F, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd_dict[i], reverse=True)
                    needed = N - len(selected_indices)
                    selected_indices.extend(sorted_front[:needed])
                    break
            
            X = R_X[selected_indices]
            F = R_F[selected_indices]

        return X
# EVOLVE-BLOCK-END
