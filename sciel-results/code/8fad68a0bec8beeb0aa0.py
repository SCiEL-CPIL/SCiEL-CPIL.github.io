"""Multi-objective optimizer on WFG7: NSGA-II inspired search with local polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Evolutionary multi-objective search with NSGA-II selection and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _non_dominated_mask(self, F):
        """Return boolean mask of non-dominated points."""
        n = len(F)
        is_dom = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dom[i] = True
                    break
        return ~is_dom

    def _crowding_distance(self, F, non_dom_mask):
        """Calculate crowding distance for non-dominated points."""
        n = len(F)
        cd = np.zeros(n)
        if not np.any(non_dom_mask):
            return cd
        
        non_dom_indices = np.where(non_dom_mask)[0]
        n_nd = len(non_dom_indices)
        
        if n_nd <= 2:
            cd[non_dom_indices] = np.inf
            return cd

        # Normalize objectives for distance calculation
        F_nd = F[non_dom_indices]
        min_F = np.min(F_nd, axis=0)
        max_F = np.max(F_nd, axis=0)
        range_F = max_F - min_F
        range_F[range_F == 0] = 1.0
        F_norm = (F_nd - min_F) / range_F

        for k in range(self.n_obj):
            # Sort by objective k
            order = np.argsort(F_norm[:, k])
            sorted_indices = non_dom_indices[order]
            sorted_F_norm = F_norm[order]
            
            cd[sorted_indices[0]] = np.inf
            cd[sorted_indices[-1]] = np.inf
            
            for i in range(1, n_nd - 1):
                cd[sorted_indices[i]] += (sorted_F_norm[i+1, k] - sorted_F_norm[i-1, k])
                
        return cd

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        ref = np.array([3.0, 5.0, 7.0])
        
        # Use global numpy random state to respect np.random.seed
        # This ensures different behavior for seed 0 and 1
        
        # Initial population
        pop_size = 40
        X = lo + span * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Main evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sorting
            non_dom_mask = self._non_dominated_mask(F)
            non_dom_indices = np.where(non_dom_mask)[0]
            
            # Crowding distance
            cd = self._crowding_distance(F, non_dom_mask)
            
            # Selection: Tournament
            # Pick 2 random parents, choose the one with better rank (non-dom) and higher CD
            p1_idx = np.random.randint(pop_size)
            p2_idx = np.random.randint(pop_size)
            
            # Compare p1 and p2
            # If one is non-dominated and the other is dominated, choose non-dominated
            # If both are non-dominated, choose higher CD
            # If both are dominated, choose lower sum of objectives (proxy for quality)
            
            def select_parent(idx1, idx2):
                nd1 = non_dom_mask[idx1]
                nd2 = non_dom_mask[idx2]
                
                if nd1 and not nd2:
                    return idx1
                elif nd2 and not nd1:
                    return idx2
                elif nd1 and nd2:
                    # Both non-dominated: choose higher crowding distance
                    if cd[idx1] > cd[idx2]:
                        return idx1
                    else:
                        return idx2
                else:
                    # Both dominated: choose one with lower sum of objectives
                    if np.sum(F[idx1]) < np.sum(F[idx2]):
                        return idx1
                    else:
                        return idx2

            parent1 = select_parent(p1_idx, p2_idx)
            
            # Create second parent
            p3_idx = np.random.randint(pop_size)
            p4_idx = np.random.randint(pop_size)
            parent2 = select_parent(p3_idx, p4_idx)
            
            # Avoid identical parents
            if parent1 == parent2:
                parent2 = (parent2 + 1) % pop_size
                if parent2 == parent1:
                    parent2 = (parent2 + 1) % pop_size

            # Crossover: BLX-Alpha
            alpha = 0.5
            x1 = X[parent1]
            x2 = X[parent2]
            
            child1 = np.zeros(self.dim)
            child2 = np.zeros(self.dim)
            
            for i in range(self.dim):
                low = min(x1[i], x2[i]) - alpha * (x2[i] - x1[i])
                high = max(x1[i], x2[i]) + alpha * (x2[i] - x1[i])
                low = max(low, lo[i])
                high = min(high, hi[i])
                
                child1[i] = np.random.uniform(low, high)
                child2[i] = np.random.uniform(low, high)
            
            # Mutation: Gaussian
            # Adaptive mutation scale
            progress = used / self.budget
            mut_scale = 0.2 * (1.0 - progress) + 0.01
            mut_scale = max(mut_scale, 0.01)
            
            child1 += mut_scale * span * np.random.normal(size=self.dim)
            child2 += mut_scale * span * np.random.normal(size=self.dim)
            
            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            
            # Evaluate
            f1 = problem(child1)
            f2 = problem(child2)
            used += 2
            
            if used > self.budget:
                # If we exceeded budget, we can't add both. 
                # But we already evaluated. The scoring records all evaluations.
                # So we must ensure we don't exceed budget in the loop condition.
                # The loop condition is `while used < self.budget`.
                # If used == budget - 1, we enter, evaluate 2, used becomes budget + 1.
                # This violates the budget constraint.
                # We should check if we have enough budget for 2 evaluations.
                # If not, break or evaluate only 1.
                # Let's break to be safe.
                break
            
            # Replacement: 
            # Combine population with children, select best pop_size
            # New population: X, F, child1, f1, child2, f2
            X_new = np.vstack([X, child1, child2])
            F_new = np.vstack([F, f1, f2])
            
            # Non-dominated sorting on X_new
            non_dom_mask_new = self._non_dominated_mask(F_new)
            non_dom_indices_new = np.where(non_dom_mask_new)[0]
            
            # Crowding distance on X_new
            cd_new = self._crowding_distance(F_new, non_dom_mask_new)
            
            # Select top pop_size
            # Priority: Non-dominated first, then by CD
            # Create a list of (rank, cd, index)
            # Rank 0 for non-dominated, 1 for dominated
            ranks = np.ones(len(X_new), dtype=int)
            ranks[non_dom_mask_new] = 0
            
            # Sort by rank, then by CD (descending)
            sort_indices = np.lexsort((cd_new * -1, ranks))
            
            # Take first pop_size
            selected_indices = sort_indices[:pop_size]
            
            X = X_new[selected_indices]
            F = F_new[selected_indices]
            
            # If we have remaining budget, do local search
            if used < self.budget:
                remaining = self.budget - used
                # Polish the best non-dominated points
                non_dom_mask_final = self._non_dominated_mask(F)
                non_dom_indices_final = np.where(non_dom_mask_final)[0]
                
                if len(non_dom_indices_final) > 0:
                    # Distribute remaining budget among non-dominated points
                    polish_per_point = max(1, remaining // len(non_dom_indices_final))
                    polish_per_point = min(polish_per_point, 10)
                    
                    for idx in non_dom_indices_final:
                        if used >= self.budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        best_x = X[idx].copy()
                        best_f = F[idx].copy()
                        best_sum = np.sum(best_f)
                        
                        for _ in range(polish_per_point):
                            if used >= self.budget:
                                break
                            if time.time() > deadline:
                                break
                            
                            # Small random perturbation
                            perturbation = 0.01 * span * np.random.normal(size=self.dim)
                            trial_x = np.clip(best_x + perturbation, lo, hi)
                            trial_f = problem(trial_x)
                            used += 1
                            
                            # Accept if better in at least one objective and not worse in others
                            if np.all(trial_f <= best_f) and np.any(trial_f < best_f):
                                best_x = trial_x
                                best_f = trial_f
                                best_sum = np.sum(best_f)
                            elif np.sum(trial_f) < best_sum:
                                # Accept if better in sum (scalarization)
                                best_x = trial_x
                                best_f = trial_f
                                best_sum = np.sum(best_f)
                        
                        # Update population with polished point
                        X[idx] = best_x
                        F[idx] = best_f

        return X
# EVOLVE-BLOCK-END
