"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid DE + Local Search Multi-Objective Optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Phase 1: Global Search using DE
        # Phase 2: Local Refinement of best non-dominated points
        
        pop_size = 30
        if self.budget < pop_size * 2:
            pop_size = max(10, self.budget // 2)
            
        # Allocate budget: 60% for global, 40% for local
        global_budget = int(self.budget * 0.6)
        local_budget = self.budget - global_budget
        
        # Ensure global_budget is at least pop_size
        if global_budget < pop_size:
            pop_size = global_budget
            local_budget = 0

        # Initialize
        X = np.random.uniform(lo, hi, size=(pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        all_F = F.copy()
        
        def fast_non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                
                if n_dom[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            
            return fronts[:-1]

        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return [np.inf] * n
            
            distances = [0.0] * n
            for m in range(self.n_obj):
                obj_vals = np.array([F[idx, m] for idx in front_indices])
                sorted_idx = np.argsort(obj_vals)
                sorted_front = [front_indices[i] for i in sorted_idx]
                
                distances[front_indices.index(sorted_front[0])] = np.inf
                distances[front_indices.index(sorted_front[-1])] = np.inf
                
                obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                if obj_range == 0:
                    continue
                
                for i in range(1, n - 1):
                    pos = front_indices.index(sorted_front[i])
                    distances[pos] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / obj_range
            
            return distances

        def de_offspring(X, F, pop_size, lo, hi, dim):
            offspring_X = []
            offspring_F = []
            
            F_factor = 0.8
            CR = 0.9
            
            for i in range(pop_size):
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                v = X[r1] + F_factor * (X[r2] - X[r3])
                
                j_rand = np.random.randint(0, dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = X[i][j]
                
                trial = np.clip(trial, lo, hi)
                
                f_trial = problem(trial)
                offspring_X.append(trial)
                offspring_F.append(f_trial)
            
            return np.array(offspring_X), np.array(offspring_F)

        # Global Search Phase
        while used < global_budget:
            if time.time() > deadline:
                break

            off_X, off_F = de_offspring(X, F, pop_size, lo, hi, self.dim)
            
            actual_off_count = min(pop_size, global_budget - used)
            off_X = off_X[:actual_off_count]
            off_F = off_F[:actual_off_count]
            
            used += actual_off_count
            
            all_F = np.vstack([all_F, off_F])
            
            combined_X = np.vstack([X, off_X])
            combined_F = np.vstack([F, off_F])
            
            fronts = fast_non_dominated_sort(combined_F)
            new_X = []
            new_F = []
            
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_front = np.argsort(-cd)
                    for i in range(remaining):
                        idx = front[sorted_front[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)

        # Local Refinement Phase
        if local_budget > 0 and len(F) > 0:
            # Get current non-dominated points
            fronts = fast_non_dominated_sort(F)
            if fronts:
                best_front = fronts[0]
                # Select top K diverse points from the best front
                K = min(10, len(best_front))
                if K > 0:
                    cd = crowding_distance(F, best_front)
                    sorted_front = np.argsort(-cd)
                    selected_indices = [best_front[i] for i in sorted_front[:K]]
                    
                    # Perform local search on these points
                    # We have local_budget evaluations left.
                    # Distribute them among K points.
                    evals_per_point = local_budget // K
                    if evals_per_point == 0:
                        evals_per_point = 1
                        K = local_budget # If budget is small, refine fewer points more? No, just 1 eval each.
                        if K > len(selected_indices):
                             selected_indices = selected_indices[:K]
                             evals_per_point = local_budget // K
                             if evals_per_point == 0:
                                 evals_per_point = 1
                                 K = local_budget

                    for idx in selected_indices:
                        if used >= self.budget:
                            break
                        if time.time() > deadline:
                            break
                            
                        x_base = X[idx]
                        f_base = F[idx]
                        
                        # Simple Random Local Search
                        for _ in range(evals_per_point):
                            if used >= self.budget:
                                break
                            if time.time() > deadline:
                                break
                                
                            # Perturb
                            step_size = 0.1 * (hi - lo)
                            x_trial = x_base + np.random.normal(0, 1, self.dim) * step_size
                            x_trial = np.clip(x_trial, lo, hi)
                            
                            f_trial = problem(x_trial)
                            used += 1
                            all_F = np.vstack([all_F, f_trial])
                            
                            # Update base if trial is better (dominates or is non-dominated and has better crowding? 
                            # For simplicity, accept if it dominates base or if base is dominated by trial)
                            # If neither dominates, keep base to maintain diversity? 
                            # Actually, for hypervolume, we want the best points. 
                            # Let's accept if trial is non-dominated by the current *global* archive? 
                            # That's expensive. 
                            # Let's just accept if trial dominates base, or if trial is better in terms of a scalar proxy (e.g. sum).
                            if np.all(f_trial <= f_base) and np.any(f_trial < f_base):
                                x_base = x_trial
                                f_base = f_trial
                            # Or, we can just record the trial and let the final archive handle it.
                            # The task records *all* returned objective vectors.
                            # So we just need to generate good points.
                            # We don't strictly need to update x_base for the next step unless we want to walk the local landscape.
                            # Let's do a simple walk: if trial is non-dominated by f_base, move to it.
                            elif not (np.all(f_base <= f_trial) and np.any(f_base < f_trial)):
                                # Neither dominates. Move to trial if it has lower sum (heuristic)
                                if np.sum(f_trial) < np.sum(f_base):
                                    x_base = x_trial
                                    f_base = f_trial

        return all_F
# EVOLVE-BLOCK-END
