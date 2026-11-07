"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Improved evolutionary multi-objective search for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)
        
        # Population size
        pop_size = min(80, self.budget)
        
        # Initialize population with some structured diversity
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        
        # Add some structured points for WFG9 (which has deceptive features)
        # WFG9 has a known structure where certain patterns in x help
        # Try some specific initializations
        structured = []
        for i in range(min(10, pop_size)):
            x_struct = np.zeros(self.dim)
            # WFG9: first 3 variables are position, rest are decision variables
            # The known Pareto front has specific structure
            x_struct[0] = lo[0] + (hi[0] - lo[0]) * (i / max(1, min(10, pop_size) - 1))
            x_struct[1] = lo[1] + (hi[1] - lo[1]) * rng.random()
            x_struct[2] = lo[2] + (hi[2] - lo[2]) * rng.random()
            # For WFG9, the last 9 variables should sum to specific values
            # Let's try setting them to create diversity
            for j in range(3, self.dim):
                x_struct[j] = lo[j] + (hi[j] - lo[j]) * rng.random()
            structured.append(x_struct)
        
        X[:len(structured)] = structured
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track archive of non-dominated points
        archive_X = []
        archive_F = []
        
        def is_dominated(f, archive):
            """Check if f is dominated by any point in archive."""
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False
        
        def dominates(f1, f2):
            """Check if f1 dominates f2."""
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        # Initialize archive
        for i in range(len(F)):
            if not is_dominated(F[i], archive_F):
                # Remove dominated points from archive
                new_archive_X = []
                new_archive_F = []
                for j in range(len(archive_F)):
                    if not dominates(F[i], archive_F[j]):
                        new_archive_X.append(archive_X[j])
                        new_archive_F.append(archive_F[j])
                archive_X.append(X[i])
                archive_F.append(F[i])
                archive_X = new_archive_X + [X[i]]
                archive_F = new_archive_F + [F[i]]
        
        def non_dominated_sort(F):
            """Return indices of non-dominated points."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]
        
        def crowding_distance(F, indices):
            """Compute crowding distance for the given indices."""
            n = len(indices)
            if n <= 2:
                return np.ones(n)
            
            distances = np.zeros(n)
            F_sub = F[indices]
            
            for m in range(self.n_obj):
                sorted_idx = np.argsort(F_sub[:, m])
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                
                min_val = F_sub[sorted_idx[0], m]
                max_val = F_sub[sorted_idx[-1], m]
                if max_val - min_val < 1e-15:
                    continue
                
                for i in range(1, n - 1):
                    distances[sorted_idx[i]] += (F_sub[sorted_idx[i+1], m] - F_sub[sorted_idx[i-1], m]) / (max_val - min_val)
            
            return distances
        
        def update_archive(child_x, child_f):
            """Update the archive with a new point."""
            nonlocal archive_X, archive_F
            if is_dominated(child_f, archive_F):
                return
            # Remove dominated points
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_F)):
                if not dominates(child_f, archive_F[i]):
                    new_archive_X.append(archive_X[i])
                    new_archive_F.append(archive_F[i])
            archive_X = new_archive_X + [child_x]
            archive_F = new_archive_F + [child_f]
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated front
            nondom_idx = non_dominated_sort(F)
            if len(nondom_idx) == 0:
                nondom_idx = np.arange(len(F))
            
            # Select parent using tournament on crowding distance
            if len(nondom_idx) > 1:
                cd = crowding_distance(F, nondom_idx)
                t1 = rng.integers(len(nondom_idx))
                t2 = rng.integers(len(nondom_idx))
                parent_idx = nondom_idx[t1] if cd[t1] >= cd[t2] else nondom_idx[t2]
            else:
                parent_idx = nondom_idx[0]
            
            parent = X[parent_idx]
            
            # Adaptive mutation
            progress = used / self.budget
            # Start with larger steps, decrease over time
            step_size = 0.3 * (1 - progress * 0.7)  # From 0.3 to 0.09
            
            # Use a mix of Gaussian and Cauchy-like mutations for diversity
            if rng.random() < 0.3:
                # Larger jump mutation
                mutation = rng.standard_normal(self.dim) * step_size * (hi - lo) * 2
            else:
                # Standard mutation
                mutation = rng.standard_normal(self.dim) * step_size * (hi - lo)
            
            child = parent + mutation
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Update archive
            update_archive(child, fc)
            
            # Replace worst point in population
            dominated_mask = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_mask[i] = True
                        break
            
            if np.any(dominated_mask):
                dominated_indices = np.where(dominated_mask)[0]
                cd_dom = crowding_distance(F, dominated_indices)
                worst_idx = dominated_indices[np.argmin(cd_dom)]
            else:
                cd_all = crowding_distance(F, np.arange(len(F)))
                worst_idx = np.argmin(cd_all)
            
            X[worst_idx], F[worst_idx] = child, fc
        
        # Final local search around best archive points
        if time.time() < deadline - 10:
            # Do a local refinement around the best archive points
            for arch_x in archive_X[:min(10, len(archive_X))]:
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                
                # Local search with smaller steps
                for _ in range(3):
                    if used >= self.budget or time.time() > deadline:
                        break
                    local_step = 0.05 * (hi - lo)
                    local_child = arch_x + rng.standard_normal(self.dim) * local_step
                    local_child = np.clip(local_child, lo, hi)
                    local_fc = problem(local_child)
                    used += 1
                    update_archive(local_child, local_fc)
        
        return np.array(archive_X) if archive_X else X
# EVOLVE-BLOCK-END
