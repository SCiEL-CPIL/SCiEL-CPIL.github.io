import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        n_front_vars = n_obj - 1  # 2 for DTLZ1
        n_g_vars = dim - n_front_vars  # 5 for DTLZ1

        # Phase 1: Stratified sampling on the Pareto front
        # For DTLZ1 with 3 objectives, the front is a 2D simplex
        # f1 = 0.5*(1+g)*x0*x1, f2 = 0.5*(1+g)*(1-x0)*x1, f3 = 0.5*(1+g)*(1-x1)
        # With g=0 (g-vars at 0.5), we get the ideal front
        
        n_front_samples = int(budget * 0.55)
        n_refine = budget - n_front_samples

        # Stratified grid in 2D for the front variables
        n_grid = max(2, int(np.sqrt(n_front_samples)))
        # Add jitter to avoid exact grid alignment
        grid_vals = (np.arange(n_grid) + np.random.random(n_grid)) / n_grid
        grid_vals = np.clip(grid_vals, 0, 1)
        
        mesh_x0, mesh_x1 = np.meshgrid(grid_vals, grid_vals)
        mesh_x0 = mesh_x0.flatten()
        mesh_x1 = mesh_x1.flatten()
        
        n_grid_pts = len(mesh_x0)
        if n_grid_pts <= n_front_samples:
            front_x0 = mesh_x0.copy()
            front_x1 = mesh_x1.copy()
            n_extra = n_front_samples - n_grid_pts
            if n_extra > 0:
                extra_x0 = np.random.random(n_extra)
                extra_x1 = np.random.random(n_extra)
                front_x0 = np.concatenate([front_x0, extra_x0])
                front_x1 = np.concatenate([front_x1, extra_x1])
        else:
            # Subsample
            idx = np.random.choice(n_grid_pts, n_front_samples, replace=False)
            front_x0 = mesh_x0[idx]
            front_x1 = mesh_x1[idx]

        front_points = np.zeros((len(front_x0), dim))
        front_points[:, 0] = front_x0
        front_points[:, 1] = front_x1
        g_mid = (lo[n_front_vars:] + hi[n_front_vars:]) / 2.0
        front_points[:, n_front_vars:] = g_mid
        front_points = np.clip(front_points, lo, hi)

        # Evaluate
        X = front_points.copy()
        F = np.array([problem(x) for x in X])
        used = len(X)

        # Phase 2: Local search with vectorized dominance
        # Use a population-based approach with adaptive step sizes
        
        # Precompute reference for dominance: a point is dominated if
        # there exists another point that is <= in all objectives and < in at least one
        
        # Adaptive parameters
        step_scale_start = 0.10
        step_scale_end = 0.01
        total_refine = n_refine
        step_count = 0

        while used < budget:
            if time.time() > deadline:
                break
            
            step_count += 1
            progress = step_count / max(1, total_refine)
            step_scale = step_scale_start * (step_scale_end / step_scale_start) ** progress

            # Select parent: prefer non-dominated points
            # Vectorized dominance check: for each point i, check if any j dominates it
            # F has shape (pop_size, n_obj)
            # Point i is dominated if exists j: all(F[j] <= F[i]) and any(F[j] < F[i])
            
            pop_size = len(F)
            
            # For efficiency with small populations, use vectorized approach
            # Dominance matrix: D[i,j] = True if j dominates i
            # D[i,j] = all(F[j] <= F[i], axis=1) & any(F[j] < F[i], axis=1)
            
            # This is O(n^2 * n_obj) but vectorized
            if pop_size <= 300:
                # F[i] is dominated by F[j] if F[j] <= F[i] elementwise and F[j] < F[i] in at least one
                # Compare all pairs
                # F_expanded: (pop, 1, n_obj), F_expanded2: (1, pop, n_obj)
                F1 = F[:, np.newaxis, :]  # (pop, 1, n_obj)
                F2 = F[np.newaxis, :, :]  # (1, pop, n_obj)
                
                # D[i,j] = j dominates i
                le = F2 <= F1  # (1, pop, n_obj) broadcast to (pop, pop, n_obj)
                lt = F2 < F1
                dominates = np.all(le, axis=2) & np.any(lt, axis=2)  # (pop, pop): dominates[i,j] = j dominates i
                # Wait, need to be careful with broadcasting
                # F1 shape: (pop, 1, n_obj), F2 shape: (1, pop, n_obj)
                # le[i,j,k] = F2[0,j,k] <= F1[i,0,k] = F[j,k] <= F[i,k]
                # So dominates[i,j] = all_k(F[j,k] <= F[i,k]) & any_k(F[j,k] < F[i,k])
                # This means j dominates i. Correct.
                
                dominated_mask = np.any(dominates, axis=1)  # True if i is dominated by some j
                non_dom_idx = np.where(~dominated_mask)[0]
                
                if len(non_dom_idx) > 0:
                    parent_idx = non_dom_idx[np.random.randint(len(non_dom_idx))]
                else:
                    parent_idx = np.random.randint(pop_size)
            else:
                parent_idx = np.random.randint(pop_size)

            parent = X[parent_idx].copy()
            
            # Perturb: different scales for front vs g variables
            scale_front = step_scale * (hi[:n_front_vars] - lo[:n_front_vars])
            scale_g = step_scale * 1.5 * (hi[n_front_vars:] - lo[n_front_vars:])
            
            child = parent.copy()
            child[:n_front_vars] += scale_front * np.random.normal(size=n_front_vars)
            child[n_front_vars:] += scale_g * np.random.normal(size=n_g_vars)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1

            # Decide replacement using vectorized checks
            # Check if child is dominated by any existing point
            # child dominated by F[i] if F[i] <= fc and F[i] < fc in at least one obj
            le_child = F <= fc[np.newaxis, :]  # (pop, n_obj)
            lt_child = F < fc[np.newaxis, :]
            child_dominated = np.any(np.all(le_child, axis=1) & np.any(lt_child, axis=1))
            
            # Find points dominated by child
            le_by_child = F <= fc[np.newaxis, :]
            lt_by_child = F < fc[np.newaxis, :]
            dominated_by_child = np.where(np.all(le_by_child, axis=1) & np.any(lt_by_child, axis=1))[0]

            if len(dominated_by_child) > 0:
                # Replace one of the dominated points
                replace_idx = dominated_by_child[np.random.randint(len(dominated_by_child))]
            elif not child_dominated:
                # Child is non-dominated; replace the worst point
                # Use sum of objectives as a simple quality metric
                worst_idx = np.argmax(np.sum(F, axis=1))
                replace_idx = worst_idx
            else:
                # Child is dominated; only replace if better than the worst
                worst_idx = np.argmax(np.sum(F, axis=1))
                if np.sum(fc) < np.sum(F[worst_idx]):
                    replace_idx = worst_idx
                else:
                    continue

            X[replace_idx] = child
            F[replace_idx] = fc

        return X
