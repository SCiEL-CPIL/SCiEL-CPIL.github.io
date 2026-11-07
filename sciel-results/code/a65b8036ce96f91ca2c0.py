"""Grid-based decomposition multi-objective optimizer for RE37."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Grid-based decomposition:
    1. Coarse grid evaluation to map the objective landscape.
    2. Identify non-dominated points on the grid.
    3. Local refinement via Nelder-Mead on promising regions.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Determine grid size based on budget
        # For dim=4: g=3 -> 81 points, g=4 -> 256 points
        # Use g=4 if budget allows (need at least ~120 for refinement)
        if budget >= 180:
            grid_points_per_dim = 4
        else:
            grid_points_per_dim = 3
        
        total_grid_points = grid_points_per_dim ** dim
        
        # If total_grid_points > budget * 0.55, reduce g
        while total_grid_points > budget * 0.55 and grid_points_per_dim > 2:
            grid_points_per_dim -= 1
            total_grid_points = grid_points_per_dim ** dim
        
        # Generate grid points
        axes = [np.linspace(lo[i], hi[i], grid_points_per_dim) for i in range(dim)]
        grid = np.array(np.meshgrid(*axes)).reshape(dim, -1).T
        
        # Evaluate grid points
        F_grid = np.array([problem(x) for x in grid])
        used = total_grid_points
        
        # Find non-dominated points in the grid
        def is_dominated(F):
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                fi = F[i]
                # Check if any other point dominates fi
                less_equal = np.all(F <= fi, axis=1)
                less_than = np.any(F < fi, axis=1)
                dominates = less_equal & less_than
                dominates[i] = False
                if np.any(dominates):
                    dominated[i] = True
            return dominated
        
        dom_grid = is_dominated(F_grid)
        non_dom_indices = np.where(~dom_grid)[0]
        
        if len(non_dom_indices) == 0:
            non_dom_indices = np.arange(len(grid))
        
        # Sort non-dominated points by their quality (sum of objectives)
        scores = np.sum(F_grid[non_dom_indices], axis=1)
        sorted_non_dom = non_dom_indices[np.argsort(scores)]
        
        remaining_budget = budget - used
        
        # Define diverse weight vectors for scalarization to cover the front
        # Use a systematic set of weights that cover the simplex
        weight_sets = []
        # Uniform weight
        weight_sets.append(np.array([1.0, 1.0, 1.0]))
        # Emphasize each objective
        weight_sets.append(np.array([2.0, 1.0, 1.0]))
        weight_sets.append(np.array([1.0, 2.0, 1.0]))
        weight_sets.append(np.array([1.0, 1.0, 2.0]))
        # Mixed weights
        weight_sets.append(np.array([1.5, 1.5, 1.0]))
        weight_sets.append(np.array([1.5, 1.0, 1.5]))
        weight_sets.append(np.array([1.0, 1.5, 1.5]))
        weight_sets.append(np.array([2.0, 1.5, 1.0]))
        weight_sets.append(np.array([1.5, 2.0, 1.0]))
        weight_sets.append(np.array([1.0, 2.0, 1.5]))
        weight_sets.append(np.array([2.0, 1.0, 1.5]))
        weight_sets.append(np.array([1.5, 1.0, 2.0]))
        weight_sets.append(np.array([1.0, 1.5, 2.0]))
        
        # Determine how many refinements we can afford
        # Each Nelder-Mead call costs ~15-25 evaluations
        # We want to spread refinements across different weight vectors
        n_weights = len(weight_sets)
        
        # Allocate budget: use ~15 evals per refinement
        evals_per_refine = 15
        max_refinements = min(n_weights, max(1, remaining_budget // evals_per_refine))
        
        # Select starting points: use the best non-dominated points
        # Map each weight to a starting point from the grid
        # Use the non-dominated point closest to the ideal point for that weight direction
        
        for i in range(max_refinements):
            if used >= budget:
                break
            
            w = weight_sets[i]
            
            # Find the grid point that best matches this weight direction
            # Score each non-dominated point by how well it aligns with the weight
            if len(sorted_non_dom) > 0:
                # Use the point with best weighted sum for this weight
                weighted_scores = np.array([np.dot(w, F_grid[idx]) for idx in sorted_non_dom])
                best_local_idx = sorted_non_dom[np.argmin(weighted_scores)]
            else:
                best_local_idx = 0
            
            x0 = grid[best_local_idx]
            
            # Define objective for Nelder-Mead: weighted sum
            def obj(x, w=w):
                nonlocal used
                if used >= budget:
                    return np.inf
                fx = problem(x)
                used += 1
                return np.dot(w, fx)
            
            # Nelder-Mead options
            maxfev = min(evals_per_refine, budget - used)
            if maxfev <= 0:
                break
            options = {'maxiter': 30, 'maxfev': maxfev, 'xatol': 1e-6, 'fatol': 1e-6}
            
            try:
                res = minimize(obj, x0, method='Nelder-Mead', 
                              bounds=list(zip(lo, hi)), options=options)
                # Evaluate the result one last time to ensure it's recorded
                if used < budget:
                    xf = res.x
                    xf = np.clip(xf, lo, hi)
                    ff = problem(xf)
                    used += 1
            except Exception:
                pass
        
        return grid
# EVOLVE-BLOCK-END
