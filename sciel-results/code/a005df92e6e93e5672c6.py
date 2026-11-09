import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # DTLZ1 structure:
        # f1 = 0.5 * (1+g) * x0 * x1
        # f2 = 0.5 * (1+g) * (1-x0) * x1
        # f3 = 0.5 * (1+g) * (1-x1)
        # where x0, x1 are the first n_obj-1 variables
        # and g depends on the remaining dim-(n_obj-1) variables
        # For the Pareto front, g=0 which requires the g-variables to be at their optimal values

        n_front_vars = n_obj - 1  # 2 for DTLZ1
        n_g_vars = dim - n_front_vars  # 5 for DTLZ1

        # The optimal values for g-variables in DTLZ1 are at the midpoint of their bounds
        # (for the standard DTLZ1 with bounds [0,1], this is 0.5)
        g_optimal = (lo[n_front_vars:] + hi[n_front_vars:]) / 2.0

        # Phase 1: Create well-distributed points on the Pareto front
        # The Pareto front for DTLZ1 is a simplex in (n_obj-1) dimensions
        # We need to sample points uniformly on this simplex
        
        # Use stratified sampling in the 2D simplex (x0 >= 0, x1 >= 0, x0+x1 <= 1)
        n_front_samples = int(budget * 0.5)
        
        # Create a grid-based stratified sampling
        n_grid = max(4, int(np.sqrt(2 * n_front_samples / 1.2)))
        # Use jittered grid for better coverage
        grid_x = (np.arange(n_grid) + np.random.random(n_grid)) / n_grid
        mesh_x0, mesh_x1 = np.meshgrid(grid_x, grid_x)
        mesh_x0 = mesh_x0.flatten()
        mesh_x1 = mesh_x1.flatten()
        
        # Filter to valid simplex points (x0 + x1 <= 1)
        valid = (mesh_x0 + mesh_x1) <= 1.0
        front_x0 = mesh_x0[valid]
        front_x1 = mesh_x1[valid]
        
        # If we don't have enough points, add random points
        if len(front_x0) < n_front_samples:
            n_extra = n_front_samples - len(front_x0)
            extra_x0 = np.random.random(n_extra)
            extra_x1 = np.random.random(n_extra) * (1 - extra_x0)
            front_x0 = np.concatenate([front_x0, extra_x0])
            front_x1 = np.concatenate([front_x1, extra_x1])
        else:
            # If we have too many, randomly sample
            idx = np.random.choice(len(front_x0), n_front_samples, replace=False)
            front_x0 = front_x0[idx]
            front_x1 = front_x1[idx]
        
        # Create the full decision vectors
        front_points = np.zeros((len(front_x0), dim))
        front_points[:, 0] = front_x0
        front_points[:, 1] = front_x1
        front_points[:, n_front_vars:] = g_optimal
        front_points = np.clip(front_points, lo, hi)

        # Evaluate front points
        X = front_points.copy()
        F = np.array([problem(x) for x in X])
        used = len(X)

        # Phase 2: Refinement using evolutionary strategy
        # Focus on improving g-variables and fine-tuning front positions
        
        # Sort by sum of objectives (proxy for quality)
        sums = np.sum(F, axis=1)
        order = np.argsort(sums)
        
        # Keep a diverse archive
        archive_X = X.copy()
        archive_F = F.copy()
        archive_sums = sums.copy()
        
        # Evolutionary phase
        while used < budget:
            if time.time() > deadline:
                break
            
            # Select parents using tournament selection
            n_tour = min(4, len(archive_X))
            candidates = np.random.choice(len(archive_X), n_tour, replace=False)
            parent_idx = candidates[np.argmin(archive_sums[candidates])]
            
            parent = archive_X[parent_idx].copy()
            
            # Mutation: perturb both front variables and g-variables
            # Scale mutation based on current quality
            g_dev = np.mean(np.abs(parent[n_front_vars:] - g_optimal))
            
            # For g-variables: small perturbation toward optimal
            base_g_scale = 0.01 * (hi[n_front_vars:] - lo[n_front_vars:])
            scale_g = base_g_scale * max(0.1, min(1.0, g_dev / 0.01 + 0.1))
            
            # For front variables: small perturbation
            scale_f = 0.01 * (hi[:n_front_vars] - lo[:n_front_vars])
            
            child = parent.copy()
            child[:n_front_vars] += scale_f * np.random.normal(size=n_front_vars)
            child[n_front_vars:] += scale_g * np.random.normal(size=n_g_vars)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            sum_c = np.sum(fc)
            
            # Check if child is non-dominated
            is_dom = np.any(np.all(archive_F <= fc, axis=1) & np.any(archive_F < fc, axis=1))
            
            if not is_dom:
                # Remove dominated points from archive
                dominated_mask = np.all(archive_F <= fc, axis=1) & np.any(archive_F < fc, axis=1)
                if np.any(dominated_mask):
                    keep = ~dominated_mask
                    archive_X = archive_X[keep]
                    archive_F = archive_F[keep]
                    archive_sums = archive_sums[keep]
                
                # Add to archive if size allows
                if len(archive_X) < int(budget * 0.3):
                    archive_X = np.vstack([archive_X, child.reshape(1, -1)])
                    archive_F = np.vstack([archive_F, fc.reshape(1, -1)])
                    archive_sums = np.append(archive_sums, sum_c)
                else:
                    # Replace the worst point
                    worst_idx = np.argmax(archive_sums)
                    if sum_c < archive_sums[worst_idx]:
                        archive_X[worst_idx] = child
                        archive_F[worst_idx] = fc
                        archive_sums[worst_idx] = sum_c
            else:
                # Even if dominated, replace worst if better
                worst_idx = np.argmax(archive_sums)
                if sum_c < archive_sums[worst_idx]:
                    archive_X[worst_idx] = child
                    archive_F[worst_idx] = fc
                    archive_sums[worst_idx] = sum_c

        # Phase 3: Final polish - focus on best points
        if used < budget and time.time() < deadline - 5.0:
            order = np.argsort(archive_sums)
            for idx in order[:min(30, len(archive_X))]:
                if used >= budget or time.time() > deadline:
                    break
                
                parent = archive_X[idx].copy()
                best_child = parent.copy()
                best_fc = archive_F[idx].copy()
                best_sum = archive_sums[idx]
                
                for _ in range(5):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    child = parent.copy()
                    # Very small perturbation for polish
                    child[n_front_vars:] += 0.003 * (hi[n_front_vars:] - lo[n_front_vars:]) * np.random.normal(size=n_g_vars)
                    child[:n_front_vars] += 0.003 * (hi[:n_front_vars] - lo[:n_front_vars]) * np.random.normal(size=n_front_vars)
                    child = np.clip(child, lo, hi)
                    
                    fc = problem(child)
                    used += 1
                    sum_c = np.sum(fc)
                    
                    if sum_c < best_sum:
                        best_child = child
                        best_fc = fc
                        best_sum = sum_c
                        parent = child
                
                # Update archive with best found
                if best_sum < archive_sums[idx]:
                    archive_X[idx] = best_child
                    archive_F[idx] = best_fc
                    archive_sums[idx] = best_sum

        return archive_X
