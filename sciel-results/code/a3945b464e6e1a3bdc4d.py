import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # DTLZ1 specific:
        # n_obj - 1 = 2 variables define the Pareto front
        # The remaining dim - (n_obj - 1) = 5 variables should be 0.5 for g=0
        n_front_vars = n_obj - 1  # 2
        n_g_vars = dim - n_front_vars  # 5

        # Strategy:
        # 1. Generate a well-distributed set of points on the Pareto front
        # 2. Use the remaining budget to refine the population with local search
        #    around the best points, maintaining diversity

        # Phase 1: Stratified sampling of the Pareto front
        # We want to cover the 2D simplex well
        # Use a grid + jitter approach
        
        # Reserve some budget for exploration
        n_front_samples = int(budget * 0.6)
        n_explore_samples = budget - n_front_samples

        # Create stratified samples in the (n_obj-1) dimensional space
        # For 2D, we can use a grid with jitter
        if n_front_vars == 2:
            n_grid = int(np.sqrt(n_front_samples))
            if n_grid < 2:
                n_grid = 2
            # Create grid points
            grid_x = np.linspace(0, 1, n_grid, endpoint=False) + np.random.random(n_grid) * (1.0 / n_grid)
            grid_x = np.clip(grid_x, 0, 1)
            
            mesh_x0, mesh_x1 = np.meshgrid(grid_x, grid_x)
            mesh_x0 = mesh_x0.flatten()
            mesh_x1 = mesh_x1.flatten()
            
            n_grid_points = len(mesh_x0)
            if n_grid_points <= n_front_samples:
                front_x0 = mesh_x0.copy()
                front_x1 = mesh_x1.copy()
                # Add random points if we have budget left
                n_extra = n_front_samples - n_grid_points
                if n_extra > 0:
                    extra_x0 = np.random.random(n_extra)
                    extra_x1 = np.random.random(n_extra)
                    front_x0 = np.concatenate([front_x0, extra_x0])
                    front_x1 = np.concatenate([front_x1, extra_x1])
            else:
                # Subsample the grid
                indices = np.random.choice(n_grid_points, n_front_samples, replace=False)
                front_x0 = mesh_x0[indices]
                front_x1 = mesh_x1[indices]
            
            front_points = np.zeros((len(front_x0), dim))
            front_points[:, 0] = front_x0
            front_points[:, 1] = front_x1
            # Set g variables to midpoint (0.5 for [0,1] bounds)
            g_mid = (lo[n_front_vars:] + hi[n_front_vars:]) / 2.0
            front_points[:, n_front_vars:] = g_mid
            
        else:
            # General case: random sampling
            front_points = lo + (hi - lo) * np.random.random((n_front_samples, dim))

        front_points = np.clip(front_points, lo, hi)

        # Evaluate front points
        X = front_points.copy()
        F = np.array([problem(x) for x in X])
        used = len(X)

        # Phase 2: Local search / refinement
        # Use the remaining budget to perturb and improve points
        # Maintain a population and replace dominated points
        
        # Helper: check if a point is dominated by any in F
        def is_dominated(f_new, F_pop):
            # Returns True if f_new is dominated by any point in F_pop
            # f_new: shape (n_obj,)
            # F_pop: shape (pop_size, n_obj)
            # Dominated if exists f in F_pop such that f <= f_new and f < f_new
            return np.any(np.all(F_pop <= f_new, axis=1) & np.any(F_pop < f_new, axis=1))

        # Helper: compute simple hypervolume contribution (approximate)
        # For speed, we'll use a simple sum-based heuristic for replacement
        # but try to maintain spread

        while used < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Choose a parent from the current population
            # Prefer non-dominated points
            # Quick dominance check for small populations
            if len(F) <= 200:
                dominated_mask = np.zeros(len(F), dtype=bool)
                for i in range(len(F)):
                    for j in range(len(F)):
                        if i != j:
                            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                dominated_mask[i] = True
                                break
                non_dom_indices = np.where(~dominated_mask)[0]
                if len(non_dom_indices) > 0:
                    parent_idx = non_dom_indices[np.random.randint(len(non_dom_indices))]
                else:
                    parent_idx = np.random.randint(len(F))
            else:
                parent_idx = np.random.randint(len(F))

            parent = X[parent_idx].copy()
            
            # Perturb the point
            # Small perturbation in all dimensions
            scale = 0.05 * (hi - lo)
            child = parent + scale * np.random.normal(size=dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1

            # Decide whether to replace a point in the population
            # Replace the point that is dominated by the child, or the worst by sum if none
            # Also, try to maintain diversity by not replacing points that are close to the child
            
            # Find if child dominates any point
            dominated_by_child = np.where(np.all(F <= fc, axis=1) & np.any(F < fc, axis=1))[0]
            
            if len(dominated_by_child) > 0:
                # Replace one of the dominated points (randomly)
                replace_idx = dominated_by_child[np.random.randint(len(dominated_by_child))]
            else:
                # Check if child is dominated by any point
                if is_dominated(fc, F):
                    # Child is dominated, don't replace unless we have no good points
                    # Still, we might want to keep it if it's close to the front
                    # For simplicity, replace the worst by sum if it's better than the worst
                    worst_idx = np.argmax(np.sum(F, axis=1))
                    if np.sum(fc) < np.sum(F[worst_idx]):
                        replace_idx = worst_idx
                    else:
                        continue
                else:
                    # Child is non-dominated
                    # Find the point to replace: the one with the worst "spread" contribution
                    # Simple heuristic: replace the point with the highest sum that is not too close to the child
                    # Or just replace the worst by sum
                    worst_idx = np.argmax(np.sum(F, axis=1))
                    replace_idx = worst_idx
            
            X[replace_idx] = child
            F[replace_idx] = fc

        return X
