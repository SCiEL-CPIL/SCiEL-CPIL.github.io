"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Multi-objective optimizer using NSGA-II with stratified initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Population size
        pop_size = 15
        if self.budget < pop_size * 2:
            pop_size = max(5, self.budget // 2)

        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=int), []
            
            ranks = np.zeros(n, dtype=int)
            fronts = []
            dominated_by = [0] * n
            dominators = [[] for _ in range(n)]
            
            for i in range(n):
                for j in range(i+1, n):
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_by[j] += 1
                        dominators[j].append(i)
                    # j dominates i?
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_by[i] += 1
                        dominators[i].append(j)
                        
            # Front 0
            front = [i for i in range(n) if dominated_by[i] == 0]
            ranks[front] = 0
            fronts.append(front)
            k = 0
            while front:
                next_front = []
                for i in front:
                    for j in dominators[i]:
                        dominated_by[j] -= 1
                        if dominated_by[j] == 0:
                            next_front.append(j)
                            ranks[j] = k + 1
                front = next_front
                if front:
                    fronts.append(front)
                k += 1
            return ranks, fronts

        # Helper: Crowding distance
        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return np.ones(len(front_indices)) * np.inf
            
            n = len(front_indices)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                obj_vals = F[front_indices, m]
                min_val = np.min(obj_vals)
                max_val = np.max(obj_vals)
                if max_val == min_val:
                    continue
                # Sort by objective m
                sorted_idx = np.argsort(obj_vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for i in range(1, n-1):
                    cd[sorted_idx[i]] += (obj_vals[sorted_idx[i+1]] - obj_vals[sorted_idx[i-1]]) / (max_val - min_val)
            return cd

        # Helper: Tournament Selection
        def tournament_select(ranks, cd, n_sel=3):
            candidates = rng.integers(0, len(ranks), size=n_sel)
            best = candidates[0]
            for c in candidates[1:]:
                # Lower rank is better
                if ranks[c] < ranks[best]:
                    best = c
                elif ranks[c] == ranks[best]:
                    # Higher crowding distance is better
                    if cd[c] > cd[best]:
                        best = c
            return best

        # Helper: Select population from a larger set
        def select_population(X_set, F_set, target_size):
            if len(F_set) <= target_size:
                return X_set, F_set
            
            ranks, fronts = non_dominated_sort(F_set)
            cd = np.zeros(len(F_set))
            for front in fronts:
                front_cd = crowding_distance(F_set, front)
                cd[front] = front_cd
                
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= target_size:
                    selected_indices.extend(front)
                else:
                    remaining = target_size - len(selected_indices)
                    # Sort front by crowding distance descending
                    front_cd_vals = cd[front]
                    sorted_front = np.argsort(front_cd_vals)[::-1]
                    selected_indices.extend(front[sorted_front[:remaining]])
                    break
                    
            return X_set[selected_indices], F_set[selected_indices]

        # Initial Phase: Stratified Sampling
        n_init = min(50, self.budget // 2)
        if n_init >= self.dim and n_init > 0:
            try:
                sampler = qmc.Sobol(d=self.dim, scramble=True)
                points = sampler.random(n_init)
                X_init = lo + (hi - lo) * points
                F_init = np.array([problem(x) for x in X_init])
                used = n_init
                X, F = select_population(X_init, F_init, pop_size)
            except Exception:
                # Fallback to random
                X = lo + (hi - lo) * rng.random((pop_size, self.dim))
                F = np.array([problem(x) for x in X])
                used = pop_size
        else:
            X = lo + (hi - lo) * rng.random((pop_size, self.dim))
            F = np.array([problem(x) for x in X])
            used = pop_size

        # Main Evolutionary Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Calculate ranks and crowding distances
            ranks, fronts = non_dominated_sort(F)
            cd = np.zeros(len(F))
            for front in fronts:
                front_cd = crowding_distance(F, front)
                cd[front] = front_cd
                
            # Select parents
            p1_idx = tournament_select(ranks, cd)
            p2_idx = tournament_select(ranks, cd)
            
            # Crossover: Uniform
            mask = rng.random(self.dim) > 0.5
            child = np.where(mask, X[p1_idx], X[p2_idx])
            
            # Mutation: Gaussian with adaptive sigma
            progress = used / self.budget
            # Start with large steps, shrink later
            sigma = (hi - lo) * (0.3 * (1 - progress) + 0.05)
            child = child + rng.standard_normal(self.dim) * sigma
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Update Population: Elitism
            # Add child to population
            X_temp = np.vstack([X, child])
            F_temp = np.vstack([F, fc])
            
            # Select best pop_size individuals
            X, F = select_population(X_temp, F_temp, pop_size)
            
        return X
# EVOLVE-BLOCK-END
