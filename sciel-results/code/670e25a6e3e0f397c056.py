"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer combining Sobol sampling, NSGA-II, and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        # Reference point for hypervolume (used for scoring)
        ref = np.array([3000.0, 0.06])
        
        # Sobol sampling for initial diverse population
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True)
        
        # Phase 1: Sobol initialization (use ~40% of budget)
        n_sobol = min(budget // 2, 100)
        # Sobol requires power of 2
        n_sobol_pow2 = 1
        while n_sobol_pow2 < n_sobol:
            n_sobol_pow2 *= 2
        n_sobol_pow2 = min(n_sobol_pow2, budget)
        
        sobol_points = sampler.random(n_sobol_pow2)
        X_sobol = lo + (hi - lo) * sobol_points[:n_sobol]
        
        F_sobol = np.array([problem(x) for x in X_sobol])
        used = n_sobol
        
        # Combine into population
        X = X_sobol.copy()
        F = F_sobol.copy()
        
        # NSGA-II for the remaining budget
        pop_size = min(50, budget - used)
        
        def non_dominated_sort(F):
            """Return fronts as list of index lists, and domination ranks."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[j] += 1
            
            current_front = [i for i in range(n) if n_dom[i] == 0]
            if not current_front:
                current_front = list(range(n))
            fronts.append(current_front)
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            
            return fronts
        
        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            n = len(indices)
            dist = np.zeros(n)
            F_sub = F[indices]
            
            for m in range(n_obj):
                min_val = F_sub[:, m].min()
                max_val = F_sub[:, m].max()
                range_val = max_val - min_val
                if range_val < 1e-15:
                    continue
                # Sort by objective m
                sorted_idx = np.argsort(F_sub[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[sorted_idx[k]] += (F_sub[sorted_idx[k + 1], m] - F_sub[sorted_idx[k - 1], m]) / range_val
            
            return dist
        
        # Main NSGA-II loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Determine population size for this generation
            gen_pop = min(pop_size, budget - used)
            if gen_pop <= 0:
                break
            
            # Create children by crossover and mutation from non-dominated parents
            fronts = non_dominated_sort(F)
            nd_indices = fronts[0] if fronts else list(range(len(F)))
            
            # Select parents
            n_parents = min(2 * gen_pop, len(nd_indices))
            parent_indices = []
            for _ in range(n_parents // 2):
                # Tournament selection on first front
                if len(nd_indices) > 1:
                    t1 = np.random.randint(0, len(nd_indices))
                    t2 = np.random.randint(0, len(nd_indices))
                    parent_indices.append(nd_indices[t1])
                    parent_indices.append(nd_indices[t2])
                else:
                    parent_indices.append(nd_indices[0])
                    parent_indices.append(nd_indices[0])
            
            # Generate children
            X_child = []
            for _ in range(gen_pop):
                if len(parent_indices) >= 2:
                    p1 = X[parent_indices[np.random.randint(0, len(parent_indices))]]
                    p2 = X[parent_indices[np.random.randint(0, len(parent_indices))]]
                    # Simulated binary crossover
                    beta = np.random.random()
                    if beta < 0.5:
                        child = beta * p1 + (1 - beta) * p2
                    else:
                        child = (1 - beta) * p1 + beta * p2
                else:
                    child = X[np.random.randint(0, len(X))].copy()
                
                # Gaussian mutation
                sigma = 0.1 * (hi - lo)
                child = child + sigma * np.random.standard_normal(dim)
                child = np.clip(child, lo, hi)
                X_child.append(child)
            
            F_child = np.array([problem(x) for x in X_child])
            used += gen_pop
            
            # Combine parent and child populations
            X_all = np.vstack([X, X_child])
            F_all = np.vstack([F, F_child])
            
            # Non-dominated sort on combined population
            fronts = non_dominated_sort(F_all)
            
            # Select next generation
            X_new = []
            F_new = []
            for front in fronts:
                if len(X_new) + len(front) <= gen_pop:
                    for idx in front:
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                else:
                    # Need to truncate this front using crowding distance
                    remaining = gen_pop - len(X_new)
                    cd = crowding_distance(F_all, front)
                    sorted_front = np.argsort(-cd)
                    for idx in front[sorted_front[:remaining]]:
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                    break
            
            if len(X_new) == 0:
                break
            X = np.array(X_new)
            F = np.array(F_new)
        
        # Phase 2: Local polish on the best points found
        if used < budget and time.time() < deadline:
            # Get the non-dominated set
            fronts = non_dominated_sort(F)
            nd_set = fronts[0] if fronts else list(range(len(F)))
            
            # Polish each non-dominated point with a few local steps
            polish_budget = min(budget - used, len(nd_set) * 5)
            if polish_budget > 0:
                per_point = max(1, polish_budget // max(1, len(nd_set)))
                for idx in nd_set:
                    if used >= budget or time.time() > deadline:
                        break
                    x = X[idx].copy()
                    f = F[idx].copy()
                    for _ in range(per_point):
                        if used >= budget or time.time() > deadline:
                            break
                        # Small random perturbation
                        x_try = x + 0.02 * (hi - lo) * np.random.standard_normal(dim)
                        x_try = np.clip(x_try, lo, hi)
                        f_try = problem(x_try)
                        used += 1
                        # Accept if it dominates or is non-dominated
                        if np.all(f_try <= f) and np.any(f_try < f):
                            x, f = x_try, f_try
                        elif not np.any(np.all(F <= f_try, axis=1) & np.any(F < f_try, axis=1)):
                            # Replace the worst point in the population
                            worst = np.argmax(np.sum(F, axis=1))
                            X[worst] = x_try
                            F[worst] = f_try
        
        return X
# EVOLVE-BLOCK-END
