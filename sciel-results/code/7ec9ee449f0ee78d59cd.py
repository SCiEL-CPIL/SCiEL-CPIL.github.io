import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We stop at 310s to ensure return.
        deadline = start_time + 310.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        max_evals = self.budget
        
        # Population size: 30 is a good balance between diversity and compute time for local search
        pop_size = 30
        
        # --- Initialization ---
        # WFG9 Pareto front is near the origin (0,0,...,0) in the decision space.
        # We initialize with a mix: 50% uniform, 50% near origin.
        X = np.zeros((pop_size, self.dim))
        
        # First half: Uniform random across the whole space (for exploration)
        X[:pop_size//2] = np.random.uniform(lo, hi, (pop_size//2, self.dim))
        
        # Second half: Biased towards the origin
        # We sample from a small box near the lower bound (origin)
        # Assuming bounds are [0, 50] or similar, 'lo' is likely 0.
        # We create points within 10-20% of the range from the lower bound.
        bias_scale = 0.2
        X[pop_size//2:] = lo + np.random.rand(pop_size//2, self.dim) * bias_scale * (hi - lo)
        
        F = np.array([problem(x) for x in X])
        evals_used = pop_size
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F_mat, n_total):
            ranks = np.zeros(n_total, dtype=int)
            S = [[] for _ in range(n_total)]
            n_dom = np.zeros(n_total, dtype=int)
            fronts = []

            for i in range(n_total):
                for j in range(n_total):
                    if i == j:
                        continue
                    if dominates(F_mat[i], F_mat[j]):
                        S[i].append(j)
                    elif dominates(F_mat[j], F_mat[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts.append([i])

            k = 0
            while k < len(fronts):
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break

            for r, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = r

            return ranks, fronts

        def crowding_distance(F_mat, indices):
            m = len(indices)
            if m <= 2:
                return np.full(m, np.inf)
            dist = np.zeros(m)
            F_sub = F_mat[indices]
            for k in range(self.n_obj):
                sorted_idx = np.argsort(F_sub[:, k])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], k] - F_sub[sorted_idx[0], k]
                if f_range > 1e-10:
                    for i in range(1, m - 1):
                        dist[sorted_idx[i]] += (F_sub[sorted_idx[i+1], k] - F_sub[sorted_idx[i-1], k]) / f_range
            return dist

        # Helper for local search: Coordinate Descent
        def local_search(x, f, budget_left):
            best_x = x.copy()
            best_f = f.copy()
            # Adaptive step size based on range
            step_sizes = 0.05 * (hi - lo)
            
            # Try a few iterations of coordinate descent
            # We limit evaluations here to save budget for global moves
            max_ls_evals = min(5, budget_left)
            evals_ls = 0
            
            for _ in range(max_ls_evals):
                if evals_ls >= max_ls_evals:
                    break
                # Pick a random dimension to perturb
                d = np.random.randint(self.dim)
                # Try positive and negative steps
                for sign in [1, -1]:
                    if evals_ls >= max_ls_evals:
                        break
                    trial_x = best_x.copy()
                    # Randomize step size slightly
                    step = step_sizes[d] * np.random.uniform(0.5, 1.5)
                    trial_x[d] = np.clip(trial_x[d] + sign * step, lo[d], hi[d])
                    
                    trial_f = problem(trial_x)
                    evals_ls += 1
                    
                    if dominates(trial_f, best_f):
                        best_x = trial_x
                        best_f = trial_f
                        break # Move to next dimension if improved
                    
            return best_x, best_f, evals_ls

        # Main Loop
        while evals_used < max_evals:
            if time.time() > deadline:
                break
            
            # Generate offspring
            # Strategy: Constrained Random Search within a shrinking hypersphere
            # Centered at the origin (lo), radius shrinks from full range to small range
            progress = evals_used / max_evals
            # Radius factor: Starts at 1.0 (full range), ends at 0.05 (5% of range)
            radius_factor = 1.0 - 0.95 * progress
            
            new_X = np.zeros((pop_size, self.dim))
            new_F = np.zeros((pop_size, self.n_obj))
            
            for i in range(pop_size):
                if evals_used >= max_evals:
                    break
                if time.time() > deadline:
                    break
                    
                # Sample within a hyper-cube centered at 'lo' (origin)
                # This ensures we stay in the region where the PF is likely to be
                # We use a uniform distribution within [lo, lo + radius_factor * (hi-lo)]
                # But to allow some exploration, we occasionally sample from the whole space?
                # No, the directive says "exclusively within a hyper-sphere... bypassing deceptive regions".
                # WFG9 deceptive regions are away from the origin. So we stick to the origin-biased box.
                
                # To ensure diversity, we scale the random vector
                # Generate random point in unit cube, scale by radius_factor
                rand_vec = np.random.rand(self.dim)
                # Map to [lo, lo + radius_factor * (hi-lo)]
                # This is effectively sampling from a sub-box near the origin
                new_X[i] = lo + rand_vec * (radius_factor * (hi - lo))
                
                # Clip just in case (though mathematically it should be within bounds if lo>=0)
                new_X[i] = np.clip(new_X[i], lo, hi)
                
                # Evaluate
                new_F[i] = problem(new_X[i])
                evals_used += 1
                
            # Combine and Select
            all_X = np.vstack([X, new_X])
            all_F = np.vstack([F, new_F])
            
            n_total = len(all_F)
            ranks, fronts = non_dominated_sort(all_F, n_total)
            
            # Select best pop_size individuals
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    remaining = pop_size - len(selected_indices)
                    dists = crowding_distance(all_F, front)
                    front_sorted = [front[i] for i in np.argsort(dists)[::-1]]
                    selected_indices.extend(front_sorted[:remaining])
                    break
                    
            X = all_X[selected_indices]
            F = all_F[selected_indices]
            
            # --- Local Search Refinement ---
            # Apply local search to the top-ranked individuals (Front 0, and maybe Front 1)
            # We identify the best individuals based on rank and crowding distance
            # For efficiency, we pick the top 3-5 individuals from the first front
            # and apply local search to them.
            
            if evals_used < max_evals - 10:
                # Get indices of the first front
                if fronts:
                    front0_indices = fronts[0]
                    # Calculate crowding for front 0 to pick the most diverse ones
                    if len(front0_indices) > 1:
                        dists = crowding_distance(F, front0_indices)
                        # Sort by crowding distance descending
                        sorted_front0 = [front0_indices[i] for i in np.argsort(dists)[::-1]]
                        # Pick top 3
                        candidates = sorted_front0[:min(3, len(sorted_front0))]
                    else:
                        candidates = front0_indices
                    
                    for idx in candidates:
                        if evals_used >= max_evals - 5:
                            break
                        if time.time() > deadline:
                            break
                        
                        # Allocate remaining budget for local search, capped at 5 evals per candidate
                        ls_budget = min(5, max_evals - evals_used)
                        if ls_budget <= 0:
                            break
                        
                        new_x, new_f, evals_used_ls = local_search(X[idx], F[idx], ls_budget)
                        evals_used += evals_used_ls
                        
                        # Update population if improved (dominates original)
                        if dominates(new_f, F[idx]):
                            X[idx] = new_x
                            F[idx] = new_f
                        # Even if not strictly dominating, if it's closer to origin or similar, 
                        # we might keep it, but strictly dominating is safer for PF approximation.
                        # However, for hypervolume, we want points on the front.
                        # If it doesn't dominate, we discard the local search result to avoid regression.

        return X
