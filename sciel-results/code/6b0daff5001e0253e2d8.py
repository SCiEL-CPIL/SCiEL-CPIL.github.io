"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Archive-based Differential Evolution."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290 # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # 1. Initialization with LHS for better coverage
        n_init = 20
        X = np.zeros((n_init, dim))
        for i in range(dim):
            perm = np.random.permutation(n_init)
            X[perm, i] = np.linspace(0, 1, n_init, endpoint=False) + np.random.uniform(0, 1/n_init, n_init)
            X[:, i] = np.clip(X[:, i], 0, 1)
        X = lo + (hi - lo) * X
        
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Archive: list of (x, f) non-dominated points
        archive_x = []
        archive_f = []
        
        def update_archive(x_new, f_new):
            # Check if new point is dominated by any existing archive point
            for i in range(len(archive_x)):
                if np.all(archive_f[i] <= f_new) and np.any(archive_f[i] < f_new):
                    return False # Dominated, do not add
            
            # Remove existing archive points dominated by new point
            keep_indices = []
            for i in range(len(archive_x)):
                if not (np.all(f_new <= archive_f[i]) and np.any(f_new < archive_f[i])):
                    keep_indices.append(i)
            
            archive_x = [archive_x[i] for i in keep_indices]
            archive_f = [archive_f[i] for i in keep_indices]
            
            # Add new point
            archive_x.append(x_new)
            archive_f.append(f_new)
            return True

        # Populate initial archive
        for i in range(n_init):
            update_archive(X[i], F[i])
            
        # DE Parameters
        F_mut = 0.8
        pop_size = 20
        
        # Main Loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Identify non-dominated points in current population for parent selection
            nd_indices = []
            for i in range(len(F)):
                dominated = False
                for j in range(len(F)):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated = True
                            break
                if not dominated:
                    nd_indices.append(i)
            
            if len(nd_indices) < 3:
                nd_indices = list(range(len(F)))
            
            # Select 3 distinct parents
            p1_idx, p2_idx, p3_idx = np.random.choice(len(nd_indices), 3, replace=False)
            p1 = X[nd_indices[p1_idx]]
            p2 = X[nd_indices[p2_idx]]
            p3 = X[nd_indices[p3_idx]]
            
            # DE Mutation
            v = p1 + F_mut * (p2 - p3)
            v = np.clip(v, lo, hi)
            
            # Evaluate
            f_v = problem(v)
            used += 1
            
            # Update Population
            # Find points in population dominated by v
            dominated_in_pop = []
            for i in range(len(F)):
                if np.all(f_v <= F[i]) and np.any(f_v < F[i]):
                    dominated_in_pop.append(i)
            
            if dominated_in_pop:
                idx_to_replace = np.random.choice(dominated_in_pop)
                X[idx_to_replace] = v
                F[idx_to_replace] = f_v
            else:
                # Check if v is dominated by any in population
                is_v_dominated = False
                for i in range(len(F)):
                    if np.all(F[i] <= f_v) and np.any(F[i] < f_v):
                        is_v_dominated = True
                        break
                
                if not is_v_dominated:
                    # v is non-dominated. Replace a random point to maintain diversity?
                    # Or replace the point with highest sum of objectives?
                    # Let's replace the point with the highest sum of objectives among non-dominated ones?
                    # Simple heuristic: Replace a random point.
                    idx_to_replace = np.random.randint(len(F))
                    X[idx_to_replace] = v
                    F[idx_to_replace] = f_v
                # If v is dominated, we don't add it to population, but we check archive.
            
            # Update Archive
            update_archive(v, f_v)

        # Local Polish Phase
        # If we have time and budget left, refine the best archive points
        if time.time() < deadline - 10 and len(archive_x) > 0:
            # Sort archive by sum of objectives (proxy for quality)
            sums = np.sum(archive_f, axis=1)
            sorted_indices = np.argsort(sums)
            
            num_polish = min(5, len(archive_x))
            polish_indices = sorted_indices[:num_polish]
            
            remaining_budget = budget - used
            if remaining_budget > 0:
                evals_per_polish = max(5, remaining_budget // (num_polish * 2))
                
                for idx in polish_indices:
                    if time.time() > deadline - 5:
                        break
                    
                    x_start = archive_x[idx]
                    f_start = archive_f[idx]
                    
                    # Simple local random search with decreasing step size
                    best_x = x_start
                    best_f = f_start
                    best_sum = np.sum(best_f)
                    
                    step_size = 0.05
                    for _ in range(evals_per_polish):
                        if time.time() > deadline - 5:
                            break
                        if used >= budget:
                            break
                            
                        # Random step
                        dx = np.random.normal(0, step_size, dim)
                        x_try = best_x + dx
                        x_try = np.clip(x_try, lo, hi)
                        
                        f_try = problem(x_try)
                        used += 1
                        
                        sum_try = np.sum(f_try)
                        
                        # Accept if better in sum (greedy) or if non-dominated?
                        # For polish, greedy on sum is okay to converge to a point.
                        if sum_try < best_sum:
                            best_x = x_try
                            best_f = f_try
                            best_sum = sum_try
                            # Decrease step size slightly?
                            step_size *= 0.9
                        else:
                            # Keep step size or decrease slowly
                            step_size *= 0.95
                        
                        # Update archive with best found so far
                        # Only update if it's non-dominated
                        # To save time, update at end of polish for this point
                        pass
                    
                    # Final update for this polished point
                    update_archive(best_x, best_f)

        return X
# EVOLVE-BLOCK-END
