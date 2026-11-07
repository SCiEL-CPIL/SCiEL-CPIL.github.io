"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using NSGA-II with adaptive crossover and mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size for NSGA-II
        pop_size = min(50, max(20, self.budget // 4))
        
        # Initialize population with Latin hypercube sampling for good spread
        X = np.zeros((pop_size, self.dim))
        for j in range(self.dim):
            perm = rng.permutation(pop_size)
            X[:, j] = lo[j] + span[j] * (perm + 0.5) / pop_size
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting returning fronts as lists of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dominated = np.zeros(n, dtype=int)  # count of points dominated by i
            fronts = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominated[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominated[i] += 1
            
            current_front = [i for i in range(n) if n_dominated[i] == 0]
            fronts.append(current_front)
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for points in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            F_sub = F[front_indices]
            
            for k in range(self.n_obj):
                order = np.argsort(F_sub[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                f_range = F_sub[order[-1], k] - F_sub[order[0], k]
                if f_range == 0:
                    continue
                for m in range(1, n - 1):
                    cd[order[m]] += (F_sub[order[m + 1], k] - F_sub[order[m - 1], k]) / f_range
            
            return cd

        def tournament_select(fronts, cd_map, k=2):
            """Tournament selection based on front rank and crowding distance."""
            candidates = rng.integers(len(fronts), size=k)
            best = candidates[0]
            for c in candidates[1:]:
                # Compare by front rank first
                if c not in cd_map:
                    continue
                if best not in cd_map:
                    best = c
                    continue
                
                # Find which front each candidate belongs to
                best_rank = next((i for i, f in enumerate(fronts) if best in f), len(fronts))
                cand_rank = next((i for i, f in enumerate(fronts) if c in f), len(fronts))
                
                if cand_rank < best_rank:
                    best = c
                elif cand_rank == best_rank:
                    # Same front, compare crowding distance
                    if cd_map.get(c, 0) > cd_map.get(best, 0):
                        best = c
            
            return best

        def sbx_crossover(x1, x2, eta_c=20.0):
            """Simulated binary crossover."""
            child1 = x1.copy()
            child2 = x2.copy()
            
            for i in range(len(x1)):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                    
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            
            return child1, child2

        def polynomial_mutation(x, eta_m=20.0):
            """Polynomial mutation."""
            x_mut = x.copy()
            for i in range(len(x)):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    
                    x_mut[i] += delta * (hi[i] - lo[i])
            
            return np.clip(x_mut, lo, hi)

        # Main NSGA-II loop
        while used < self.budget:
            # Compute fronts and crowding distances
            fronts = fast_non_dominated_sort(F)
            cd_map = {}
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, front)
                    for idx, dist in zip(front, cd):
                        cd_map[idx] = dist
            
            # Create offspring population
            offspring_X = []
            offspring_F = []
            
            # Determine how many offspring we can create this generation
            max_offspring = min(pop_size, self.budget - used)
            
            for _ in range(max_offspring):
                # Select two parents via tournament
                p1 = tournament_select(fronts, cd_map)
                p2 = tournament_select(fronts, cd_map)
                
                # Crossover
                if rng.random() < 0.9:  # Crossover probability
                    c1, c2 = sbx_crossover(X[p1], X[p2], eta_c=20.0)
                else:
                    c1 = X[p1].copy()
                    c2 = X[p2].copy()
                
                # Mutation
                c1 = polynomial_mutation(c1, eta_m=20.0)
                c2 = polynomial_mutation(c2, eta_m=20.0)
                
                # Evaluate offspring
                f1 = problem(c1)
                f2 = problem(c2)
                used += 2
                
                offspring_X.append(c1)
                offspring_F.append(f1)
                offspring_X.append(c2)
                offspring_F.append(f2)
                
                if used >= self.budget:
                    break
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring populations
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            combined_n = len(combined_F)
            
            # Non-dominated sort on combined population
            combined_fronts = fast_non_dominated_sort(combined_F)
            combined_cd = {}
            for front in combined_fronts:
                if len(front) > 0:
                    cd = crowding_distance(combined_F, front)
                    for idx, dist in zip(front, cd):
                        combined_cd[idx] = dist
            
            # Select next generation
            new_X = []
            new_F = []
            remaining = pop_size
            
            for front in combined_fronts:
                if len(front) <= remaining:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    remaining -= len(front)
                else:
                    # Need to select from this front based on crowding distance
                    front_cd = [(idx, combined_cd.get(idx, 0)) for idx in front]
                    front_cd.sort(key=lambda x: x[1], reverse=True)
                    for idx, _ in front_cd[:remaining]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    remaining = 0
                    break
            
            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)
            
            if used >= self.budget:
                break

        return X
# EVOLVE-BLOCK-END
